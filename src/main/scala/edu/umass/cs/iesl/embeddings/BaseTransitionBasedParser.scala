package edu.umass.cs.iesl.embeddings

import cc.factorie.app.nlp._
import cc.factorie._
import cc.factorie.app.nlp.pos.PennPosTag
import parse.WSJTransitionBasedParser
import scala.collection.mutable.{HashMap, ArrayBuffer}
import scala.util.parsing.json.JSON
import scala.annotation.tailrec
import java.io._
import cc.factorie.util.BinarySerializer
import cc.factorie.util.{BinarySerializer, FileUtils}
import scala._
import cc.factorie.optimize._
import scala.concurrent.Await
import cc.factorie.variable.LabeledCategoricalVariable
import cc.factorie.variable.BinaryFeatureVectorVariable
import cc.factorie.variable.CategoricalVectorDomain
import cc.factorie.variable.{LabeledCategoricalVariable, BinaryFeatureVectorVariable, CategoricalVectorDomain, CategoricalDomain}
import scala.collection.mutable
import cc.factorie.app.classify.backend._
import scala.Some
import scala.Some
import cc.factorie.app.nlp.DocumentAnnotator
import cc.factorie.variable.CategoricalDomain
import cc.factorie.app.classify.backend.LinearMulticlassClassifier
import cc.factorie.app.classify.backend.MulticlassClassifierTrainer
import cc.factorie.app.nlp.Sentence
import cc.factorie.optimize.AdaGradRDA
import cc.factorie.app.classify.backend.OnlineLinearMulticlassTrainer
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.pos.PennPosTag
import cc.factorie.app.nlp.lemma
import cc.factorie.app.nlp.Token
import cc.factorie.app.nlp.SharedNLPCmdOptions
import cc.factorie.util.FileUtils
import cc.factorie.app.nlp.load
import cc.factorie.app.classify.backend.SVMMulticlassTrainer
import cc.factorie.optimize.OptimizableObjectives
import cc.factorie.app.nlp.DocumentAnnotator
import scala.Some
import scala.Some
import cc.factorie.app.nlp.parse._





/** Default transition-based dependency parser. */
abstract class BaseTransitionBasedParser extends DocumentAnnotator {

  def serialize(stream: java.io.OutputStream): Unit
  def deserialize(stream: java.io.InputStream): Unit

  case class ParseDecision(action: String) {
    val Array(lrnS, srpS, label) = action.split(" ")
    val leftOrRightOrNo = lrnS.toInt 		// leftarc-rightarc-noarc
    val shiftOrReduceOrPass = srpS.toInt	// shift-reduce-pass
  }

  object labelDomain extends CategoricalDomain[String]
  val defaultCategory = "-1 -1 N"
  labelDomain += defaultCategory

  class ParseDecisionVariable(targetDecision: ParseDecision, val state: ParseState) extends LabeledCategoricalVariable(targetDecision.action) {
    def domain = labelDomain
    val features = new NonProjDependencyParserFeatures(this)
    val stackWord = state.stackToken(0).form
    val inputWord = state.inputToken(0).form

    /* Include <NULL>s */
    //    featureGenerators.foreach(f => features += f.apply(state))

    /* DO NOT include <NULL>s */
    // TODO if we want to keep this in here, change implementation to use Option instead of <NULL> string?
    featureGenerators.foreach(f => {
      val featString = f.apply(state)
      if("<NULL>".r.findAllIn(featString).length-1 != "\\|".r.findAllIn(featString).length) features += featString
    })
  }
  object featuresDomain extends CategoricalVectorDomain[String]
  class NonProjDependencyParserFeatures(val decisionVariable: ParseDecisionVariable) extends BinaryFeatureVectorVariable[String] {
    def domain = featuresDomain
    override def skipNonCategories = domain.dimensionDomain.frozen

    /* remove bias for now */
    //this += "BIAS"
  }

  // Serialization
  def serialize(file: File): Unit = {
    if (file.getParentFile ne null) file.getParentFile.mkdirs()
    serialize(new java.io.FileOutputStream(file))
  }
  def deserialize(file: File): Unit = {
    require(file.exists(), "Trying to load non-existent file: '" +file)
    deserialize(new java.io.FileInputStream(file))
  }



  val parseDecisionCache = collection.mutable.HashMap[String,ParseDecision]()
  def getParseDecision(s: String): ParseDecision = parseDecisionCache.getOrElseUpdate(s, new ParseDecision(s))



  def testString(testSentences:Iterable[Sentence]): String = {
    val(las, uas, tokSpeed, sentSpeed) = test(testSentences)
    "LAS="+las+" UAS="+uas+s"  ${tokSpeed} tokens/sec"
  }

  def test(testSentences:Iterable[Sentence]): (Double, Double, Double, Double) = {
    val t0 = System.currentTimeMillis()
    testSentences.par.foreach(process)
    val totalTime = System.currentTimeMillis() - t0
    val totalTokens = testSentences.map(_.tokens.length).sum
    val totalSentences = testSentences.size
    val pred = testSentences.map(_.attr[ParseTree])
    (ParserEval.calcLas(pred), ParserEval.calcUas(pred), totalTokens*1000.0/totalTime, totalSentences*1000.0/totalTime)
  }


  lazy val testFeatureSpec = io.Source.fromURL(this.getClass.getResource("/parser-features.json")).getLines().mkString("\n")
  lazy val featureGenerators: Seq[DependencyFeatures.DependencyFeatureGenerator] = DependencyFeatures.fromJSON(testFeatureSpec)

  object ParserConstants {
    val ROOT_ID = 0

    val SHIFT  = 1
    val REDUCE = 2
    val PASS   = 3

    val LEFT  = 4
    val RIGHT = 5
    val NO    = 6

    val TRAINING   = 7
    val PREDICTING = 8
    val BOOSTING   = 9

    def getString(constantVal: Int): String = constantVal match {

      case ParserConstants.SHIFT => "shift"
      case ParserConstants.REDUCE => "reduce"
      case ParserConstants.PASS => "pass"

      case ParserConstants.LEFT => "left"
      case ParserConstants.RIGHT => "right"
      case ParserConstants.NO => "no"

      case ParserConstants.TRAINING => "training"
      case ParserConstants.PREDICTING => "predicting"
      case ParserConstants.BOOSTING => "boosting"

      case ParserConstants.ROOT_ID => "root id"

      case _ => throw new Error(s"Integer value $constantVal is not defined in ParserConstants")
    }
  }
  def classify(pd: ParseDecisionVariable ):  ParseDecision

  def generateDecisions(ss: Iterable[Sentence], mode: Int, nThreads: Int): Iterable[ParseDecisionVariable] = {
    val decs = cc.factorie.util.Threading.parMap(ss, nThreads)(s => {
      val oracle: NonProjectiveOracle = {
        if (mode == ParserConstants.TRAINING) new NonprojectiveGoldOracle(s)
        else new NonprojectiveBoostingOracle(s, classify)
      }
      new NonProjectiveShiftReduce(oracle.predict).parse(s)
      oracle.instances.toSeq
    })
    decs.flatten
  }


  // For DocumentAnnotator trait
  def process(doc: Document) = { doc.sentences.foreach(process); doc }
  def prereqAttrs = Seq(classOf[Sentence], classOf[PennPosTag], classOf[lemma.WordNetTokenLemma]) // Sentence also includes Token
  def postAttrs = Seq(classOf[ParseTree])
  override def tokenAnnotationString(token:Token): String = {
    val sentence = token.sentence
    val pt = if (sentence ne null) sentence.attr[ParseTree] else null
    if (pt eq null) "_\t_"
    else (pt.parentIndex(token.positionInSentence)+1).toString+"\t"+pt.label(token.positionInSentence).categoryValue
  }
  //override def tokenAnnotationString(token:Token): String = { val parse = token.parseParent; if (parse ne null) parse.positionInSentence+"\t"+token.parseLabel.categoryValue else "_\t_" }

  def process(s: Sentence): Sentence = {
    val parse = s.attr.getOrElseUpdate(new ParseTree(s))
    new NonProjectiveShiftReduce(predict = classify).parse(s).zipWithIndex.map(dt => {
      parse.setParent(dt._2, dt._1._1)
      parse.label(dt._2).set(ParseTreeLabelDomain.index(dt._1._2))(null)
    })
    s
  }

  /* Takes features and turns them into a parse decision using predict(ParseDecisionVariable => ParseDecision) */
  val defaultDecision = ParseDecision(defaultCategory)
  class NonProjectiveShiftReduce(val predict: ParseDecisionVariable => ParseDecision) {
    import ParserConstants._
    def parse(s: Sentence) = {
      // ParseState(lambda, beta, reduceID, sentence)
      val state = new ParseState(0, 1, collection.mutable.HashSet[Int](), s)

      while(state.input < state.sentenceTokens.length) {
        if (state.stack < 0)
          noShift(state)
        else {
          val decision = new ParseDecisionVariable(defaultDecision, state)
          val label = predict(decision)
          val beta = state.inputToken(0)
          val lambda = state.stackToken(0)

          /* Debugging output */
          //          println(s"${ParserConstants.getString(label.leftOrRightOrNo)} ${ParserConstants.getString(label.shiftOrReduceOrPass)} ${label.label}, lambda: ${lambda.form}, beta: ${beta.form}")
          //          println(s"lambda: form=${lambda.form}, head=${if (lambda.hasHead) lambda.head.depToken.form else "<NULL>"}, head2=${if (lambda.hasGrandHead) lambda.grandHead.depToken.form else "<NULL>"}, lmd=${if (lambda.form != "<NULL>") lambda.leftmostDependent.form else "<NULL>"}, lmd2=${if (lambda.form != "<NULL>") lambda.leftmostDependent2.form else "<NULL>"}, rmd=${if (lambda.form != "<NULL>") lambda.rightmostDependent.form else "<NULL>"}, rmd2=${if (lambda.form != "<NULL>") lambda.rightmostDependent2.form else "<NULL>"}")
          //          println(s"beta: form=${beta.form}, head=${if (beta.hasHead) beta.head.depToken.form else "<NULL>"}, head2=${if (beta.hasGrandHead) beta.grandHead.depToken.form else "<NULL>"}, lmd=${if (beta.form != "<NULL>") beta.leftmostDependent.form else "<NULL>"}, lmd2=${if (beta.form != "<NULL>") beta.leftmostDependent2.form else "<NULL>"}, rmd=${if (beta.form != "<NULL>") beta.rightmostDependent.form else "<NULL>"}, rmd2=${if (beta.form != "<NULL>") beta.rightmostDependent2.form else "<NULL>"}")
          //          println()

          if (label.leftOrRightOrNo == LEFT) {
            if (state.stack == ROOT_ID) noShift(state)
            else if (beta.isDescendentOf(lambda)) noPass(state)
            else if (label.shiftOrReduceOrPass == REDUCE) leftReduce(label.label, state)
            else leftPass(label.label, state)
          }
          else if (label.leftOrRightOrNo == RIGHT) {
            if (lambda.isDescendentOf(beta)) noPass(state)
            else if (label.shiftOrReduceOrPass == SHIFT) rightShift(label.label, state)
            else rightPass(label.label, state)
          }
          else {
            if (label.shiftOrReduceOrPass == SHIFT) noShift(state)
            else if (label.shiftOrReduceOrPass == REDUCE && lambda.hasHead) noReduce(state)
            else noPass(state)
          }
        }
      }
      state.sentenceTokens.drop(1).map(dt => if (dt.hasHead) (dt.head.depToken.thisIdx-1, dt.head.label) else (-1,""))
    }

    private def passAux(state: ParseState): Unit = {
      var i = state.stack - 1
      while (i >= 0) {
        if (!state.reducedIds.contains(i)) {
          state.stack = i
          return
        }
        i -= 1
      }
      state.stack = i
    }

    private def leftArc(label: String, state: ParseState)  { state.stackToken(0).setHead(state.inputToken(0), label) }
    private def rightArc(label: String, state: ParseState) { state.inputToken(0).setHead(state.stackToken(0), label) }

    private def shift(state: ParseState)  { state.stack = state.input; state.input += 1 }
    private def reduce(state: ParseState) { state.reducedIds.add(state.stack); passAux(state) }
    private def pass(state: ParseState)   { passAux(state: ParseState) }

    private def noShift(state: ParseState)  { shift(state) }
    private def noReduce(state: ParseState) { reduce(state) }
    private def noPass(state: ParseState)   { pass(state) }
    private def leftReduce(label: String, state: ParseState) { leftArc(label, state);  reduce(state) }
    private def leftPass(label: String, state: ParseState)   { leftArc(label, state);  pass(state)   }
    private def rightShift(label: String, state: ParseState) { rightArc(label, state); shift(state)  }
    private def rightPass(label: String, state: ParseState)  { rightArc(label, state); pass(state)   }
  }

  trait NonProjectiveOracle {
    import ParserConstants._
    val sentence: Sentence
    def predict(state: ParseDecisionVariable): ParseDecision

    var instances = new ArrayBuffer[ParseDecisionVariable] { override val initialSize = 100 }
    def getSimpleDepArcs = sentence.parse.targetParents.map(_ + 1).zip(sentence.parse.labels.map(_.target.value.category))
    def getDepArcs = { Seq((-1, "<ROOT-ROOT>")) ++ getSimpleDepArcs.map { case (i: Int, l: String) => (i, l) } }
    val goldHeads = getDepArcs

    def getGoldDecision(state: ParseState): ParseDecision = {
      val shiftOrReduceOrPass =
        getGoldLRN(state) match {
          case LEFT  => if (shouldGoldReduce(hasHead=true, state=state)) REDUCE else PASS
          case RIGHT => if (shouldGoldShift(state=state)) SHIFT else PASS
          case _ => {
            if (shouldGoldShift(state=state)) SHIFT
            else if (shouldGoldReduce(hasHead=false, state=state)) REDUCE
            else PASS
          }
        }
      new ParseDecision(getGoldLRN(state) + " " + shiftOrReduceOrPass + " " + getGoldLabel(state))
    }

    def getGoldLabel(state: ParseState): String = {
      if (goldHeads(state.stack)._1 == state.input) goldHeads(state.stack)._2
      else if (goldHeads(state.input)._1 == state.stack) goldHeads(state.input)._2
      else "N"
    }

    def getGoldLRN(state: ParseState): Int = {
      if (goldHeads(state.stack)._1 == state.input) LEFT
      else if (goldHeads(state.input)._1 == state.stack) RIGHT
      else NO
    }

    def shouldGoldShift(state: ParseState): Boolean = {
      if (goldHeads(state.input)._1 < state.stack) return false
      else
        for (i <- (state.stack - 1) until 0 by -1) if (!state.reducedIds.contains(i)) {
          if (goldHeads(i)._1 == state.input)
            return false
        }
      true
    }

    def shouldGoldReduce(hasHead: Boolean, state: ParseState): Boolean = {
      if (!hasHead && !state.stackToken(0).hasHead)
        return false
      for (i <- (state.input + 1) until state.sentenceTokens.length)
        if (goldHeads(i)._1 == state.stack)
          return false
      true
    }
  }

  class NonprojectiveGoldOracle(val sentence: Sentence) extends NonProjectiveOracle {
    def predict(decisionVariable: ParseDecisionVariable): ParseDecision = {
      val decision = getGoldDecision(decisionVariable.state)
      instances += new ParseDecisionVariable(decision, decisionVariable.state)
      decision
    }
  }

  class NonprojectiveBoostingOracle(val sentence: Sentence, basePredict: ParseDecisionVariable => ParseDecision) extends NonProjectiveOracle {
    def predict(decisionVariable: ParseDecisionVariable): ParseDecision = {
      val label = new ParseDecisionVariable(getGoldDecision(decisionVariable.state), decisionVariable.state)
      instances += label
      basePredict(label)
    }
  }

  object DependencyFeatures {
    val locationAbbrevs = collection.mutable.HashMap(
      "S_LAMBDA" -> "l",
      "S_STACK"  -> "s",
      "S_BETA"   -> "b",
      "R_H"      -> "h",     // head
      "R_LMD"    -> "lmd",   // left-most dependent
      "R_RMD"    -> "rmd"    // right-most dependent
    )
    val formAbbrevs = collection.mutable.HashMap(
      "F_FORM"   -> "f",
      "F_LEMMA"  -> "m",
      "F_POS"    -> "p",
      "F_DEPREL" -> "d",
      "F_LNPL"   -> "lnpl", // left-nearest punctuation of lambda
      "F_RNPL"   -> "rnpl", // right-nearest punctuation of lambda
      "F_LNPB"   -> "lnpb", // left-nearest punctuation of beta
      "F_RNPB"   -> "rnpb"  // right-nearest punctuation of beta
    )
    val locationFns: HashMap[String, (Int) => (ParseState) => DepToken] = mutable.HashMap(
      "b"   -> ((offset: Int) => (state: ParseState) => state.inputToken(offset)),
      "l"   -> ((offset: Int) => (state: ParseState) => state.lambdaToken(offset)),
      "s"   -> ((offset: Int) => (state: ParseState) => state.stackToken(offset)),
      "l_h" -> ((_: Int) => (state: ParseState) => if (state.lambdaToken(0).hasHead) state.lambdaToken(0).head.depToken else null),
      "l_lmd" -> ((offset: Int) => (state: ParseState) => state.lambdaToken(offset).leftmostDependent),
      "l_rmd" -> ((offset: Int) => (state: ParseState) => state.lambdaToken(offset).rightmostDependent),
      "b_lmd" -> ((offset: Int) => (state: ParseState) =>  state.stackToken(offset).leftmostDependent),
      "b_rmd" -> ((offset: Int) => (state: ParseState) =>  state.stackToken(offset).rightmostDependent),

      // left-nearest sibling of stack
      "l_lns" -> ((offset: Int) => (state: ParseState) =>  state.lambdaToken(offset).leftNearestSibling),

      /* 3rd order features */

      // grand-head of lambda
      "l_h2" -> ((_: Int) => (state: ParseState) => if (state.lambdaToken(0).hasGrandHead) state.lambdaToken(0).grandHead.depToken else null),

      // 2nd left-most dependent of lambda
      "l_lmd2" -> ((offset: Int) => (state: ParseState) => state.lambdaToken(offset).leftmostDependent2),

      // 2nd right-most dependent of lambda
      "l_rmd2" -> ((offset: Int) => (state: ParseState) => state.lambdaToken(offset).rightmostDependent2),

      // 2nd left-most dependent of beta
      "b_lmd2" -> ((offset: Int) => (state: ParseState) =>  state.stackToken(offset).leftmostDependent2)
    )

    // TODO make this nicer
    val formFns = HashMap(
      "f"   -> ((t: DepToken) => "f:" + (if (t != null) t.form else "<NULL>")),
      "m"   -> ((t: DepToken) => "m:" + (if (t != null) t.lemma else "<NULL>")),
      "p"   -> ((t: DepToken) => "p:" + (if (t != null) t.pos else "<NULL>")),
      "d"   -> ((t: DepToken) => "d:" + (if (t != null && t.hasHead) t.head.label else "<NULL>")),
      "b0"  -> ((t: DepToken) => "lFirst:" + (if (t != null && t.thisIdx != -1) t.state.lambdaToken(0) eq t.state.sentenceTokens(1) else false).toString),
      "b1"  -> ((t: DepToken) => "bLast:" + (if (t != null) t.state.stackToken(0) eq t.state.sentenceTokens.last else false).toString),
      "b2"  -> ((t: DepToken) => "adjacent:" + (if (t != null) t.state.input - t.state.stack == 1 else false).toString)
    )

    /* Takes a string definition of a feature template and applies it to a ParseState to get
     * the string feature for the given ParseState */
    def generators(locationOffsetAndForm: String): (ParseState => String) = {
      val LocationOffsetAndForm = """([a-z_]*)[+]?([-0-9]*):([a-z]*[0-9]?)""".r
      locationOffsetAndForm match {
        case LocationOffsetAndForm(location, offset, form) => {
          val locationFn = locationFns(location)(if (offset == "") 0 else offset.toInt)
          (state: ParseState) => location + offset + ":" + formFns(form)(locationFn(state))
        }
        case _ => throw new Error("Couldn't parse location and form from feature generator string.")
      }
    }

    abstract class DependencyFeatureGenerator extends (ParseState => String)
    class SingletonDependencyFeatureGenerator(f: String) extends DependencyFeatureGenerator {
      lazy val featureFn = generators(f)
      def apply(s: ParseState): String = featureFn(s)
    }
    class CompositeDependencyFeatureGenerator(gens: Seq[DependencyFeatureGenerator]) extends DependencyFeatureGenerator {
      def apply(s: ParseState) = gens.map(_.apply(s)).mkString("|")
    }
    private def stripJSONComments(s: String) = s.split("\n").map(_.split("#").head).mkString("\n")
    def fromJSON(source: String) = {
      val someJson = JSON.parseFull(stripJSONComments(source))
      val featureSpec = someJson match {
        case map: Some[Map[String, List[List[String]]] @unchecked] => map.get("features")
        case _ => throw new Error()
      }
      featureSpec.map(fs => {
        val fGens = fs.map(f => new SingletonDependencyFeatureGenerator(f))
        if (fGens.length > 1) new CompositeDependencyFeatureGenerator(fGens)
        else fGens.head
      })
    }
  }

  class DepToken(val form: String, val lemma: String, val pos: String, val thisIdx: Int, val state: ParseState) {
    var head: DepArc = null
    def grandHead: DepArc = if(hasHead) head.depToken.head else null
    def hasHead: Boolean = head != null
    def hasGrandHead: Boolean = grandHead != null

    def setHead(headToken: DepToken, label: String) {
      head = new DepArc(headToken, label)

      if(head.depToken.thisIdx != -1){
        // set left and rightmost dependencies
        if (thisIdx < head.depToken.thisIdx)
          state.leftmostDeps(head.depToken.thisIdx) = thisIdx
        else
          state.rightmostDeps(head.depToken.thisIdx) = thisIdx
      }
    }

    def leftmostDependent: DepToken = {
      if (thisIdx == -1) state.nullToken
      else{
        val i = state.leftmostDeps(thisIdx)
        if (i == -1) state.nullToken
        else state.sentenceTokens(i)
      }
    }

    def rightmostDependent: DepToken = {
      if (thisIdx == -1) state.nullToken
      else{
        val i = state.rightmostDeps(thisIdx)
        if (i == -1) state.nullToken
        else state.sentenceTokens(i)
      }
    }

    def leftmostDependent2: DepToken = {
      if (thisIdx == -1) state.nullToken
      else{
        val i = state.leftmostDeps(thisIdx)
        if (i == -1) state.nullToken
        else{
          val j = state.leftmostDeps(i)
          if (j == -1) state.nullToken
          else state.sentenceTokens(j)
        }
      }
    }
    def rightmostDependent2: DepToken = {
      if (thisIdx == -1) state.nullToken
      else{
        val i = state.rightmostDeps(thisIdx)
        if (i == -1) state.nullToken
        else{
          val j = state.rightmostDeps(i)
          if (j == -1) state.nullToken
          else state.sentenceTokens(j)
        }
      }
    }

    def leftNearestSibling: DepToken = {
      if(hasHead){
        var i = thisIdx - 1
        var sib = state.nullToken
        while(i >= 0 && sib == state.nullToken){
          if (state.sentenceTokens(i).hasHead && state.sentenceTokens(i).head.depToken == head.depToken)
            sib = state.sentenceTokens(i)
          i -= 1
        }
        sib
      }
      else state.nullToken
    }

    def rightNearestSibling: DepToken = {
      if(hasHead){
        var i = thisIdx + 1
        var sib = state.nullToken
        while(i < state.sentenceTokens.size && sib == state.nullToken){
          if (state.sentenceTokens(i).hasHead && state.sentenceTokens(i).head.depToken == head.depToken)
            sib = state.sentenceTokens(i)
          i += 1
        }
        sib
      }
      else state.nullToken
    }

    @tailrec final def isDescendentOf(that: DepToken): Boolean = {
      if (!hasHead) false
      else if (this.head.depToken == that) true
      else this.head.depToken.isDescendentOf(that)
    }
  }

  case class DepArc(depToken: DepToken, label: String)

  class ParseState(var stack: Int, var input: Int, val reducedIds: collection.mutable.HashSet[Int], sentence: Sentence) {
    private def depToken(token: Token, idx: Int, state: ParseState) = new DepToken(form = token.string, lemma = token.lemmaString, pos = token.posTag.categoryValue, thisIdx=idx, state=state)
    val rootToken = new DepToken(form = "<ROOT>",  lemma = "<ROOT>", pos = "<ROOT>", thisIdx = 0, state=this)
    val nullToken = new DepToken(form = "<NULL>",  lemma = "<NULL>", pos = "<NULL>", thisIdx = -1, state=this)

    val sentenceTokens = (Seq(rootToken) ++ sentence.tokens.zipWithIndex.map(t => depToken(t._1, t._2+1, this))).toArray

    val leftmostDeps = Array.fill[Int](sentenceTokens.size)(-1)
    val rightmostDeps = Array.fill[Int](sentenceTokens.size)(-1)

    def inputToken(offset: Int): DepToken = {
      val i = input + offset
      if (i < 0 || sentenceTokens.size - 1 < i) nullToken
      else sentenceTokens(i)
    }

    def lambdaToken(offset: Int): DepToken = {
      val i = stack + offset
      if (i < 0 || sentenceTokens.size - 1 < i) nullToken
      else sentenceTokens(i)
    }

    def stackToken(offset: Int): DepToken = {
      if (offset == 0)
        return sentenceTokens(stack)

      var off = math.abs(offset)
      var dir = if (offset < 0) -1 else 1
      var i = stack + dir
      while (0 < i && i < input) {
        if (!reducedIds.contains(i)) {
          off -= 1
          if (off == 0)
            return sentenceTokens(i)
        }
        i += dir
      }
      nullToken
    }
  }
}

