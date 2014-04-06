package edu.umass.cs.iesl.embeddings

import collection.mutable
import cc.factorie.la.{DenseTensor1, DenseTensor2, Tensor}

class MatlabInterop(fn: String){

  val matrixReader = new com.jmatio.io.MatFileReader(fn)

  def getTensor2(key: String): DenseTensor2 =  mlArrayToTensor2(matrixReader.getMLArray(key))
  def getArrayOfTensor1(key: String): Array[DenseTensor1] = mlArrayToArrayOfTensor1(matrixReader.getMLArray(key))

  def mlArrayToTensor2(m: com.jmatio.types.MLArray): DenseTensor2 = {
    val rows = m.getM()
    val cols = m.getN()
    val x = m.asInstanceOf[com.jmatio.types.MLDouble].getArray
    val t = new DenseTensor2(rows,cols)
    var i = 0;
    while(i < rows){
      var j = 0
      while(j < cols){
        t(i,j) = x(i)(j)
        j +=1
      }
      i+=1
    }
    t
  }
  def mlArrayToArrayOfTensor1(m: com.jmatio.types.MLArray): Array[DenseTensor1] = {
    val rows = m.getM()
    val cols = m.getN()
    val x = m.asInstanceOf[com.jmatio.types.MLDouble].getArray
    val t = new DenseTensor2(rows,cols)
    var i = 0;
    while(i < rows){
      var j = 0
      while(j < cols){
        t(i,j) = x(i)(j)
        j +=1
      }
      i+=1
    }
    t
  }
}

