package edu.umass.cs.iesl.embeddings

import cc.factorie.la.DenseTensor1
import java.io._
import java.util.zip.{GZIPInputStream, GZIPOutputStream}


object EmbeddingSerialization {
   def serialize(f: String, data: Seq[DenseTensor1]): Unit = {
     val writer = new DataOutputStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(f))))
     writer.write(data.head.dim1)
     writer.write(data.length)
     data.foreach(d =>   d.foreach(dd => writer.writeDouble(dd)))
     writer.flush()
     writer.close
   }
   def deserialize(f: String): Array[DenseTensor1]= {
    val reader = new DataInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))))
    val latentDim = reader.readInt()
    val numEmbeddings = reader.readInt()
    val arr = Array.fill[DenseTensor1](numEmbeddings)(new DenseTensor1(latentDim))
    var i = 0; var j = 0
    while(i < numEmbeddings){
      while(j < latentDim){
        arr(i)(j) = reader.readDouble()
        j +=1
      }
      i +=1
    }
    arr
  }
}
