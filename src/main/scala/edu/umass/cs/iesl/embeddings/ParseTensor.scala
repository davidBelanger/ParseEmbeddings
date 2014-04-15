package edu.umass.cs.iesl.embeddings

import cc.factorie.la.{DenseTensor1, DenseTensor2}
import cc.factorie.util.BinarySerializer
import collection.immutable.HashMap
import java.io.File



trait ParseTensor{
  val wordDomain = collection.mutable.HashMap[String,Int]()
  val arcDomain = collection.mutable.HashMap[String,Int]()

  def getScoresForPair(childWord: String, parentWord: String) : collection.Map[String,Double]
  def getWordIndex(w: String): Int

}

class KruskalParseTensor(tensorFilename: String, domainFilename: String, numTake: Int = -1) extends ParseTensor{
  //load the tensor
  println("loading the parse tensor")
  val childWeights = EmbeddingSerialization.deserialize(tensorFilename + ".rows",numTake)
  val parentWeights = EmbeddingSerialization.deserialize(tensorFilename + ".cols",numTake)
  val arcWeights = EmbeddingSerialization.deserialize(tensorFilename + ".arcs",numTake)

  val latentDim = childWeights(0).length

  BinarySerializer.deserialize(wordDomain,new File(domainFilename + ".words"))
  BinarySerializer.deserialize(arcDomain,new File(domainFilename + ".arcs"))

  println("there are " + wordDomain.size + " words in the word domain")
  println("there are " + arcDomain.size  + " arcs in the arc domain  ")
  assert(wordDomain.contains("OOV"))
  val rootIndex = wordDomain("root")
  wordDomain += "<root>" -> rootIndex
  val oovIndex = wordDomain("OOV")
  val numWordsInTensor = childWeights.length
  def getWordIndex(w: String): Int = wordDomain.get(w).filter(_ < numWordsInTensor).getOrElse(oovIndex)


  //todo: maybe make different failure modes for the case both words are OOV v.s. one is OOV
  def getScoresForPair(childWord: String, parentWord: String) : collection.Map[String,Double]  = {
    val childIndex = getWordIndex(childWord)
    val parentIndex = getWordIndex(parentWord)
    arcDomain.mapValues(getScoresForArc(childIndex,parentIndex,_))
  }

  def getScoresForArc(childIndex: Int, parentIndex: Int, arcIndex: Int) : Double = {
    val child = childWeights(childIndex)
    val parent = parentWeights(parentIndex)
    val arc = arcWeights(arcIndex)
    TensorUtil.threeWayDotProd(child,parent,arc)

  }


}

object TensorUtil{
  def threeWayDotProd(a: DenseTensor1, b: DenseTensor1, c: DenseTensor1): Double = {
    var i = 0
    var sum = 0.0
    val aa = a.asArray
    val bb = b.asArray
    val cc = c.asArray
    val n = a.length
    while(i < n){
      sum += aa(i)*bb(i)*cc(i)
      i += 1
    }
    sum
  }
  def  zipVectors(a: DenseTensor1,b: DenseTensor1): DenseTensor1 = {
    val c = new DenseTensor1(a.length)
    (0 until c.length).foreach(i => c(i) = a(i)*b(i))
    c
  }

}

trait ParseTensorOptions extends cc.factorie.util.CmdOptions  {
  val useTensor = new CmdOption("use-parse-tensor",false,"BOOLEAN","Whether to use word embeddings")
  val tensorFile = new CmdOption("tensor-file", "", "STRING", "path to parsetensor .mat file")
  val tensorDomainFile = new CmdOption("tensor-domain-file","","STRING","where the word->Int domains are kept for the parsetensor")
}