package edu.umass.cs.iesl.embeddings

import cc.factorie.la.DenseTensor1
import java.io._
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import cc.factorie.la
import cc.factorie.variable.CategoricalDomain
import collection.immutable.HashMap


object EmbeddingSerialization {
   def serialize(f: String, data: Seq[DenseTensor1]): Unit = {
     val writer = new DataOutputStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(f))))
     writer.writeInt(data.head.dim1)
     writer.writeInt(data.length)
     data.foreach(d =>   d.foreach(dd => writer.writeDouble(dd)))
     writer.flush()
     writer.close
   }
   def deserialize(f: String,numTake: Int = -1): Array[DenseTensor1]= {
     println("reading from "  + f)
    val reader = new DataInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))))

    val latentDim = reader.readInt()
    val numEmbeddings = reader.readInt()
     println("latent dim = " + latentDim)
    val arr = Array.fill[DenseTensor1](numEmbeddings)(new DenseTensor1(latentDim))
    var i = 0; var j = 0
    val numEmbeddingsToTake = if (numTake > 0) numTake else numEmbeddings
     println("num embeddings to load = " + numEmbeddingsToTake + " (out of " + numEmbeddings + ")")

     while(i < numEmbeddingsToTake){
      while(j < latentDim){
        arr(i)(j) = reader.readDouble()
        j +=1
      }
      if (i % 5000 == 0 && i > 0) println("loaded " + i + " vectors")
      i +=1
    }
    arr
  }
}


class WordEmbeddingFromBinary(embeddingFile: String, wordDomainFile: String,numTake: Int = -1) extends scala.collection.mutable.LinkedHashMap[String,la.DenseTensor1] {
  var dimensionSize = 0
  initialize()
  def initialize() {
      val tensors = EmbeddingSerialization.deserialize(embeddingFile,numTake)
    println("loaded vectors")
      dimensionSize = tensors.head.length
     val numLoaded = tensors.length
    println("loading domain")
      val domain = collection.mutable.HashMap[String,Int]()
      cc.factorie.util.BinarySerializer.deserialize(domain,wordDomainFile)
      domain.iterator.foreach(si => {
        val string = si._1
        val idx = si._2
        if(idx < numLoaded) this += string -> tensors(idx)
      })
    println("embedding deserialization complete")

  }

}
