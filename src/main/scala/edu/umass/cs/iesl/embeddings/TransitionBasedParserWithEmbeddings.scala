package edu.umass.cs.iesl.embeddings

import cc.factorie.app.nlp._
import cc.factorie._
import app.classify.backend.OnlineLinearMulticlassTrainer
import embeddings.{WordEmbeddingOptions, WordEmbedding}
import java.io._
import cc.factorie.util._
import parse.{ParseTree, ParserEval}
import scala._
import cc.factorie.optimize._
import scala.concurrent.Await
import cc.factorie.app.nlp.Sentence
import cc.factorie.optimize.AdaGradRDA
import cc.factorie.app.nlp.load
import cc.factorie.optimize.OptimizableObjectives
import cc.factorie.la.{GrowableSparseBinaryTensor1}
import cc.factorie.DenseTensor1
import variable.CategoricalDomain

class TransitionBasedParserWithWordEmbeddings(embedding: WordEmbedding) extends TransitionBasedParserWithEmbeddings{
  val denseFeatureDomainSize = labelDomain.size
  val defaultEmbedding = new DenseTensor1(denseFeatureDomainSize)
  def getEmbedding(str: String) : DenseTensor1 = {
    if (embedding == null)
      defaultEmbedding
    else {
      if(embedding.contains(str))
        embedding(str)
      else
        defaultEmbedding
    }
  }

  def getDenseFeaturesFromStrings(w1: String, w2: String): DenseTensor1 = {
    new DenseTensor1(labelDomain.size,getEmbedding(w1).dot(getEmbedding(w2)))
  }
}

class TransitionBasedParserWithParseEmbeddings(tensor: ParseTensor) extends TransitionBasedParserWithEmbeddings{
  val tensorScoreIsSymmetric = false
  def getDenseFeaturesFromStrings(w1: String, w2: String): DenseTensor1 = {
    if(tensorScoreIsSymmetric)
      getDenseFeaturesFromStringsSymmetric(w1,w2)
    else
      getDenseFeaturesFromStringsAssymmetric(w1,w2)
  }
  def getDenseFeaturesFromStringsSymmetric(w1: String, w2: String): DenseTensor1 = {
    val output = new DenseTensor1(labelDomain.size)

    val string2score = tensor.getScoresForPair(w1,w2)
    (0 until labelDomain.size).foreach( i=> {
      val Array(_, _, label) = labelDomain.category(i).split(" ")

      val score = if(label == "N") 0.0 else string2score(label)
      output(i) = score
    })


    //todo for debugging . remove
//    println("\n\n"+w1 + " " + w2)
//    (0 until labelDomain.size).foreach( i=> {
//      val Array(_, _, label) = labelDomain.category(i).split(" ")
//      println(label + " " + output(i))
//    })
    output
  }

  def getDenseFeaturesFromStringsAssymmetric(w1: String, w2: String): DenseTensor1 = {
    val output = new DenseTensor1(labelDomain.size)

    val string2score_child2parent = tensor.getScoresForPair(w1,w2)
    val string2score_parent2child = tensor.getScoresForPair(w1,w2)

    (0 until labelDomain.size).foreach( i=> {
      val Array(lrnS, srpS, label) = labelDomain.category(i).split(" ")        //todo: precompute these things
      val leftOrRightOrNo = lrnS.toInt 		// leftarc-rightarc-noarc
      val child2parent = leftOrRightOrNo ==  ParserConstants.RIGHT //todo: check this

      val score = if(label == "N") 0.0 else {
         if(child2parent)
           string2score_child2parent(label)
        else
           string2score_parent2child(label)
      }
      output(i) = score
    })
    output
  }
}

abstract class TransitionBasedParserWithEmbeddings extends BaseTransitionBasedParser {

  def serialize(stream: java.io.OutputStream): Unit = {
    import cc.factorie.util.CubbieConversions._
    // Sparsify the evidence weights
    import scala.language.reflectiveCalls
    val sparseEvidenceWeights = new la.DenseLayeredTensor2(featuresDomain.dimensionDomain.size, labelDomain.size, new la.SparseIndexedTensor1(_))
    model.weightsForSparseFeatures.value.foreachElement((i, v) => if (v != 0.0) sparseEvidenceWeights += (i, v))
    model.weightsForSparseFeatures.set(sparseEvidenceWeights)
    val dstream = new java.io.DataOutputStream(new BufferedOutputStream(stream))
    BinarySerializer.serialize(featuresDomain.dimensionDomain, dstream)
    BinarySerializer.serialize(labelDomain, dstream)
    BinarySerializer.serialize(model, dstream)
    dstream.close()
  }
  def deserialize(stream: java.io.InputStream): Unit = {
    import cc.factorie.util.CubbieConversions._
    // Get ready to read sparse evidence weights
    val dstream = new java.io.DataInputStream(new BufferedInputStream(stream))
    BinarySerializer.deserialize(featuresDomain.dimensionDomain, dstream)
    BinarySerializer.deserialize(labelDomain, dstream)
    import scala.language.reflectiveCalls
    model.weightsForSparseFeatures.set(new la.DenseLayeredTensor2(featuresDomain.dimensionDomain.size, labelDomain.size, new la.SparseIndexedTensor1(_)))
    BinarySerializer.deserialize(model, dstream)
    println("TransitionBasedParser model parameters oneNorm "+model.parameters.oneNorm)
    dstream.close()
  }


  def getFeatures(v: ParseDecisionVariable): (GrowableSparseBinaryTensor1,DenseTensor1) =  {
    val denseFeatures =  getDenseFeatures(v)
    (v.features.value.asInstanceOf[GrowableSparseBinaryTensor1],denseFeatures)
  }

  def getDenseFeatures(v: ParseDecisionVariable): DenseTensor1 = {
    getDenseFeaturesFromStrings(v.state.stackToken(0).form, v.state.inputToken(0).form)
  }
  def getDenseFeaturesFromStrings(w1: String, w2: String): DenseTensor1

  lazy val model = new SparseAndDenseClassConditionalLinearMulticlassClassifier[GrowableSparseBinaryTensor1,DenseTensor1](labelDomain.size, featuresDomain.dimensionSize)

  def classify(v: ParseDecisionVariable) = getParseDecision(labelDomain.category(model.classification(getFeatures(v)).bestLabelIndex))
  def trainFromVariables(vs: Iterable[ParseDecisionVariable], trainer: Trainer, objective: OptimizableObjectives.Multiclass,evaluate: (SparseAndDenseClassConditionalLinearMulticlassClassifier[_,_]) => Unit) {
    val examples = vs.map(v => {
      val features = getFeatures(v)
      new PredictorExample(model, features, v.target.intValue, objective, 1.0)
    })

    val rand = new scala.util.Random(0)
    (0 until 3).foreach(_ => {
      trainer.trainFromExamples(examples.shuffle(rand))
      evaluate(model)
    })
  }
  def boosting(ss: Iterable[Sentence], nThreads: Int, trainer: Trainer, objective: OptimizableObjectives.Multiclass,evaluate: SparseAndDenseClassConditionalLinearMulticlassClassifier[_,_] => Unit) =
    trainFromVariables(generateDecisions(ss, ParserConstants.BOOSTING, nThreads), trainer, objective,evaluate)

}


class TransitionBasedParserWithEmbeddingsArgs extends TransitionBasedParserArgs with WordEmbeddingOptions with ParseTensorOptions

object TransitionBasedParserWithParseTensorTrainer extends TransitionBasedParserWithEmbeddingsTrainer{
  def evaluateParameters(args: Array[String]) = {
    val opts = new TransitionBasedParserWithEmbeddingsArgs
    opts.parse(args)
    assert(! (opts.useEmbeddings.value && opts.useTensor.value),"can't specify to use both word embeddings and parsetensor embeddings")
    assert(opts.useTensor.value,"use TransitionBasedParserWithWordEmbeddingsTrainer if you want to use word embeddings or no embeddings")
    val tensor = new KruskalParseTensor(opts.tensorFile.value,opts.tensorDomainFile.value)

    val newModelFactory = () =>  new TransitionBasedParserWithParseEmbeddings(tensor)
    evaluateParametersFromModel(newModelFactory,args)

  }
}

object TransitionBasedParserWithWordEmbeddingsTrainer extends TransitionBasedParserWithEmbeddingsTrainer{
  def evaluateParameters(args: Array[String]) = {
    val opts = new TransitionBasedParserWithEmbeddingsArgs
    opts.parse(args)
    assert(! (opts.useEmbeddings.value && opts.useTensor.value),"can't specify to use both word embeddings and parsetensor embeddings")

    val useEmbeddings = opts.useEmbeddings.value
    if(useEmbeddings) println("using embeddings") else println("not using embeddings")
    val embedding = if(useEmbeddings)  new WordEmbedding(() => new FileInputStream(opts.embeddingFile.value),opts.embeddingDim.value,opts.numEmbeddingsToTake.value) else null

    val newModelFactory = () =>  new TransitionBasedParserWithWordEmbeddings(embedding)
    evaluateParametersFromModel(newModelFactory,args)

  }
}



abstract class TransitionBasedParserWithEmbeddingsTrainer extends cc.factorie.util.HyperparameterMain {
  def evaluateParametersFromModel(newModelFactory: () => TransitionBasedParserWithEmbeddings,args: Array[String]) = {
    val c =  newModelFactory()

    val opts = new TransitionBasedParserWithEmbeddingsArgs
    implicit val random = new scala.util.Random(0)
    opts.parse(args)

    assert(opts.trainFiles.wasInvoked || opts.trainDir.wasInvoked)
    val objective = OptimizableObjectives.hingeMulticlass

    // Load the sentences
    def loadSentences(listOpt: opts.CmdOption[List[String]], dirOpt: opts.CmdOption[String]): Seq[Sentence] = {
      var fileList = Seq.empty[String]
      if (listOpt.wasInvoked) fileList = listOpt.value.toSeq
      if (dirOpt.wasInvoked) fileList ++= FileUtils.getFileListFromDir(dirOpt.value)
      fileList.flatMap(fname => {
        if(opts.wsj.value)
          load.LoadWSJMalt.fromFilename(fname, loadLemma=load.AnnotationTypes.AUTO, loadPos=load.AnnotationTypes.AUTO).head.sentences.toSeq
        else if (opts.ontonotes.value)
          load.LoadOntonotes5.fromFilename(fname, loadLemma=load.AnnotationTypes.AUTO, loadPos=load.AnnotationTypes.AUTO).head.sentences.toSeq
        else
          load.LoadConll2008.fromFilename(fname).head.sentences.toSeq
      })
    }

    val sentencesFull = loadSentences(opts.trainFiles, opts.trainDir)
    val devSentencesFull = loadSentences(opts.devFiles, opts.devDir)
    val testSentencesFull = loadSentences(opts.testFiles, opts.testDir)

    val trainPortionToTake = if(opts.trainPortion.wasInvoked) opts.trainPortion.value.toDouble  else 1.0
    val testPortionToTake =  if(opts.testPortion.wasInvoked) opts.testPortion.value.toDouble  else 1.0
    val sentences = sentencesFull.take((trainPortionToTake*sentencesFull.length).floor.toInt)
    val testSentences = testSentencesFull.take((testPortionToTake*testSentencesFull.length).floor.toInt)
    val devSentences = devSentencesFull.take((testPortionToTake*devSentencesFull.length).floor.toInt)

    println("Total train sentences: " + sentences.size)
    println("Total test sentences: " + testSentences.size)

    def testSingle(c: BaseTransitionBasedParser, ss: Seq[Sentence], extraText: String = ""): Unit = {
      if (ss.nonEmpty) {
        println(extraText + " " + c.testString(ss))
      }
    }

    def testAll(c: BaseTransitionBasedParser, extraText: String = ""): Unit = {
      println("\n")
      testSingle(c, sentences,     "Train " + extraText)
      testSingle(c, devSentences,  "Dev "   + extraText)
      testSingle(c, testSentences, "Test "  + extraText)
    }

    // Load other parameters
    val numBootstrappingIterations = opts.bootstrapping.value.toInt

    val l1 = 2*opts.l1.value / sentences.length
    val l2 = 2*opts.l2.value / sentences.length
    val optimizer = new AdaGradRDA(opts.rate.value, opts.delta.value, l1, l2)
    println(s"Initializing trainer (${opts.nTrainingThreads.value} threads)")


//    def evaluate(cls: SparseAndDenseBilinearMulticlassClassifier[_,_,_]) {
//      println(cls.weightsForSparseFeatures.value.toSeq.count(x => x == 0).toFloat/cls.weightsForSparseFeatures.value.length +" sparsity")
//      testAll(c, "iteration ")
//    }

    def evaluate(cls: SparseAndDenseClassConditionalLinearMulticlassClassifier[_,_]) {
      //println(cls.weightsForSparseFeatures.value.toSeq.count(x => x == 0).toFloat/cls.weightsForSparseFeatures.value.length +" sparsity")
      testAll(c, "iteration ")
    }

    c.featuresDomain.dimensionDomain.gatherCounts = true
    println("Generating decisions...")
    c.generateDecisions(sentences, c.ParserConstants.TRAINING, opts.nFeatureThreads.value)

    println("Before pruning # features " + c.featuresDomain.dimensionDomain.size)
    c.featuresDomain.dimensionDomain.trimBelowCount(2*opts.cutoff.value)
    c.featuresDomain.freeze()
    c.featuresDomain.dimensionDomain.gatherCounts = false
    println("After pruning # features " + c.featuresDomain.dimensionDomain.size)

    println("Getting Decisions...")

    var trainingVs = c.generateDecisions(sentences, c.ParserConstants.TRAINING, opts.nFeatureThreads.value)

    println("Training...")

    val trainer = new OnlineTrainer(c.model.parameters,optimizer,maxIterations = opts.maxIters.value)
    c.trainFromVariables(trainingVs, trainer, objective, evaluate)

//    val trainer =  new OnlineLinearMulticlassTrainer(optimizer=optimizer, useParallel=if (opts.nTrainingThreads.value > 1) true else false, nThreads=opts.nTrainingThreads.value, objective=OptimizableObjectives.hingeMulticlass, maxIterations=opts.maxIters.value)
//    c.trainFromVariables(trainingVs,trainer,objective,evaluate)
//

    trainingVs = null // GC the old training labels
//    for (i <- 0 until numBootstrappingIterations) {
//      println("Boosting iteration " + i)
//      c.boosting(sentences, nThreads=opts.nTrainingThreads.value, trainer=trainer, evaluate=evaluate)
//    }

    //testSentences.foreach(c.process)

    testSingle(c, testSentences, "")

    val testLAS = ParserEval.calcLas(testSentences.map(_.attr[ParseTree]))
    if(opts.targetAccuracy.wasInvoked) cc.factorie.assertMinimalAccuracy(testLAS,opts.targetAccuracy.value.toDouble)
    if (opts.saveModel.value) {
      val modelUrl: String = if (opts.modelDir.wasInvoked) opts.modelDir.value else opts.modelDir.defaultValue + System.currentTimeMillis().toString + ".factorie"
      c.serialize(new java.io.File(modelUrl))
      val d = newModelFactory()
      d.deserialize(new java.io.File(modelUrl))
      testSingle(d, testSentences, "Post serialization accuracy ")
    }

    testLAS
  }
}


//object TransitionBasedParserWithEmbeddingsTester {
//  def main(args: Array[String]) {
//    val opts = new TransitionBasedParserWithEmbeddingsArgs
//    opts.parse(args)
//    assert(opts.testDir.wasInvoked || opts.testFiles.wasInvoked)
//
//    // load model from file if given,
//    // else if the wsj command line param was specified use wsj model,
//    // otherwise ontonotes model
//    val parser = {
//      val useEmbeddings = opts.useEmbeddings.value
//      if(useEmbeddings) println("using embeddings") else println("not using embeddings")
//      val embedding = if(useEmbeddings)  new WordEmbedding(() => new FileInputStream(opts.embeddingFile.value),opts.embeddingDim.value,opts.numEmbeddingsToTake.value) else null
//      val parser = new TransitionBasedParserWithEmbeddings(embedding)
//      parser.deserialize(new File(opts.modelDir.value))
//      parser
//    }
//
//    assert(!(opts.testDir.wasInvoked && opts.testFiles.wasInvoked))
//    val testFileList = if(opts.testDir.wasInvoked) FileUtils.getFileListFromDir(opts.testDir.value) else opts.testFiles.value.toSeq
//
//    val testPortionToTake =  if(opts.testPortion.wasInvoked) opts.testPortion.value else 1.0
//    val testDocs =  testFileList.map(fname => {
//      if(opts.wsj.value)
//        load.LoadWSJMalt.fromFilename(fname, loadLemma=load.AnnotationTypes.AUTO, loadPos=load.AnnotationTypes.AUTO).head
//      else
//        load.LoadOntonotes5.fromFilename(fname, loadLemma=load.AnnotationTypes.AUTO, loadPos=load.AnnotationTypes.AUTO).head
//    })
//    val testSentencesFull = testDocs.flatMap(_.sentences)
//    val testSentences = testSentencesFull.take((testPortionToTake*testSentencesFull.length).floor.toInt)
//
//    println(parser.testString(testSentences))
//  }
//}


object TransitionBasedParserWithEmbeddingsOptimizer {
  def main(args: Array[String]) {
    val opts = new TransitionBasedParserWithEmbeddingsArgs
    opts.parse(args)
    opts.saveModel.setValue(false)
    val l1 = cc.factorie.util.HyperParameter(opts.l1, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val l2 = cc.factorie.util.HyperParameter(opts.l2, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val rate = cc.factorie.util.HyperParameter(opts.rate, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val delta = cc.factorie.util.HyperParameter(opts.delta, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val cutoff = cc.factorie.util.HyperParameter(opts.cutoff, new cc.factorie.util.SampleFromSeq[Int](Seq(0, 1, 2)))
    val bootstrap = cc.factorie.util.HyperParameter(opts.bootstrapping, new cc.factorie.util.SampleFromSeq[Int](Seq(0, 1, 2)))
    val maxit = cc.factorie.util.HyperParameter(opts.maxIters, new cc.factorie.util.SampleFromSeq[Int](Seq(2, 5, 8)))
    /*
    val ssh = new cc.factorie.util.SSHActorExecutor("apassos",
      Seq("avon1", "avon2"),
      "/home/apassos/canvas/factorie-test",
      "try-log/",
      "cc.factorie.app.nlp.parse.TransitionBasedParser",
      10, 5)
      */
    val qs = new cc.factorie.util.QSubExecutor(32, "cc.factorie.app.nlp.parse.TransitionBasedParserWithEmbeddingsTrainer")
    val optimizer = new cc.factorie.util.HyperParameterSearcher(opts, Seq(l1, l2, rate, delta, cutoff, bootstrap, maxit), qs.execute, 200, 180, 60)
    val result = optimizer.optimize()
    println("Got results: " + result.mkString(" "))
    opts.saveModel.setValue(true)
    println("Running best configuration...")
    import scala.concurrent.duration._
    Await.result(qs.execute(opts.values.flatMap(_.unParse).toArray), 2.hours)
    println("Done")
  }
}


class TransitionBasedParserArgs extends cc.factorie.util.DefaultCmdOptions with SharedNLPCmdOptions{
  val trainFiles =  new CmdOption("train", Nil.asInstanceOf[List[String]], "FILENAME...", "")
  val testFiles =  new CmdOption("test", Nil.asInstanceOf[List[String]], "FILENAME...", "")
  val trainDir = new CmdOption("trainDir", "", "FILENAME", "Directory containing training files.")
  val testDir = new CmdOption("testDir", "", "FILENAME", "Directory containing test files.")
  val devDir = new CmdOption("devDir", "", "FILENAME", "Directory containing dev files.")
  val devFiles =   new CmdOption("dev", Nil.asInstanceOf[List[String]], "FILENAME...", "")
  val ontonotes = new CmdOption("onto", true, "BOOLEAN", "Whether data are in Ontonotes format or otherwise (WSJ or CoNLL)")
  val wsj = new CmdOption("wsj", false, "BOOLEAN", "Whether data are in WSJ format or otherwise (Ontonotes or CoNLL)")
  val cutoff    = new CmdOption("cutoff", 0, "", "")
  val loadModel = new CmdOption("load", "", "", "")
  val nTrainingThreads =  new CmdOption("num-training-threads", 1, "INT", "How many threads to use during training.")
  val nFeatureThreads =  new CmdOption("num-feature-threads", 1, "INT", "How many threads to use during training.")

  val useSVM =    new CmdOption("use-svm", false, "BOOL", "Whether to use SVMs to train")
  val modelDir =  new CmdOption("model", "model", "FILENAME", "File in which to save the trained model.")
  val bootstrapping = new CmdOption("bootstrap", 0, "INT", "The number of bootstrapping iterations to do. 0 means no bootstrapping.")
  val saveModel = new CmdOption("save-model", true,"BOOLEAN","whether to write out a model file or not")
  val l1 = new CmdOption("l1", 0.000001,"FLOAT","l1 regularization weight")
  val l2 = new CmdOption("l2", 0.00001,"FLOAT","l2 regularization weight")
  val rate = new CmdOption("rate", 10.0,"FLOAT","base learning rate")
  val maxIters = new CmdOption("max-iterations", 5, "INT", "iterations of training per round")
  val delta = new CmdOption("delta", 100.0,"FLOAT","learning rate decay")
}