package edu.umass.cs.iesl.embeddings

import collection.mutable
import cc.factorie.la.{DenseTensor1, DenseTensor2, Tensor}
import com.jmatio.types.{MLArray, MLInt64, MLDouble}
import collection.mutable.ArrayBuffer
import java.io.File
import java.util

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
    val t = Array.fill[DenseTensor1](rows)(new DenseTensor1(cols))
    var i = 0;
    while(i < rows){
      var j = 0
      while(j < cols){
        t(i)(j) = x(i)(j)
        j +=1
      }
      i+=1
    }
    t
  }
}

object Factorie2Matlab{

//  def writeArray2ToMat(t: Array[Int], numRows: Int,f: String,name: String): Unit = {
//    val data = new MLInt64( name, t, numRows)
//    new com.jmatio.io.MatFileWriter(f,data)
//  }

  def writeArray2ToMat(t: Array[Double], numRows: Int,f: String,name: String): Unit = {
    val data = new MLDouble( name, t, numRows)

    val list = new util.ArrayList[MLArray]()
    list.add(data)

    new com.jmatio.io.MatFileWriter(f,list)
  }

  def writeTensor2ToMat(t: DenseTensor2, f: String,name: String): Unit = {
    writeArray2ToMat(t.asArray,t.dim1,f,name)
  }
}

object CSV2MatFile{
  def main(args: Array[String]) {
    val in = args(0)
    val delim = args(1)
    val out = args(2)
    val name = args(3)
    val data = new ArrayBuffer[Int]()
    val reader = io.Source.fromFile(in).getLines()
    val firstLine = reader.next().map(_.toInt)
    val len = firstLine.length
    data ++= firstLine
    data ++= reader.flatMap(_.split(delim)).map(_.toInt)
    val numRows = data.length/len
    Factorie2Matlab.writeArray2ToMat(data.toArray.map(_.toDouble),numRows,out,name)    //todo: keep these as ints
  }
}



