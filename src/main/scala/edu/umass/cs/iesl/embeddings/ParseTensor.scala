package edu.umass.cs.iesl.embeddings

import cc.factorie.la.{DenseTensor1, DenseTensor2}


class ParseTensor(filename: String) {
  val matlab = new MatlabInterop(filename)
  val childWeights = matlab.getArrayOfTensor1("child")
  val parentWeights = matlab.getArrayOfTensor1("parent")
  val arcWeights = matlab.getTensor2("arc")
  val arcWeightsArray = matlab.getArrayOfTensor1("arc")

  def getScoresForArcs(childIndex: Int, parentIndex: Int) : DenseTensor1 = {
     val child = childWeights(childIndex)
     val parent = parentWeights(parentIndex)
     arcWeights.leftMultiply(child).dot(parent)
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
