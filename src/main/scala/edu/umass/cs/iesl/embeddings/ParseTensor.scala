package edu.umass.cs.iesl.embeddings

import cc.factorie.la.{DenseTensor1, DenseTensor2}
import cc.factorie.util.BinarySerializer
import collection.immutable.HashMap


class ParseTensor(tensorFilename: String, domainFilename: String) {
  //load the tensor
  println("loading the parse tensor")
  val matlab = new MatlabInterop(filename)
  val childWeights = matlab.getArrayOfTensor1("child")
  val parentWeights = matlab.getArrayOfTensor1("parent")
  val arcWeights = matlab.getTensor2("arc")
  val arcWeightsArray = matlab.getArrayOfTensor1("arc")
  println("loading the string -> int domain for the parse tensor")
  val wordDomain = collection.mutable.HashMap[String,Int]()
  val arcDomain = collection.mutable.HashMap[String,Int]()

  BinarySerializer.deserialize(domainFilename,wordDomain,arcDomain)
  println("there are " + wordDomain.size + " words in the word domain")
  println("there are " + arcDomain.size  + " arcs in the arc domain  ")
  assert(wordDomain.contains("OOV"))
  val oovIndex = wordDomain("OOV")

  def getWordIndex(w: String): Int = wordDomain.getOrElse(w,oovIndex)
  //todo: need to unpack this to have the right order w.r.t the current domain
  def getScoresForPair(childWord: String, parentWord: String): DenseTensor1  = {
    val childIndex = getWordIndex(childWord)
    val parentIndex = getWordIndex(parentWord)
    getScoresForPair(childIndex,parentIndex)
  }

  def getScoresForPair(childIndex: Int, parentIndex: Int) : DenseTensor1 = {
//     val child = childWeights(childIndex)
//     val parent = parentWeights(parentIndex)
//     val r = arcWeights.leftMultiply(child)
//      r.dot(parent)
//       .dot(parent)
    ???
  }

  def getScoresForArc(childIndex: Int, parentIndex: Int, arcIndex: Int) : Double = {
    val child = childWeights(childIndex)
    val parent = parentWeights(parentIndex)
    val arc = arcWeightsArray(arcIndex)
    threeWayDotProd(child,parent,arc)

  }
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

}
