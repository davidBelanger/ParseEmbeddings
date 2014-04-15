package edu.umass.cs.iesl.embeddings

import cc.factorie.optimize.{OptimizableObjectives, UnivariateOptimizableObjective}
import cc.factorie.la.DenseTensor1
import util.Random
import cc.factorie.util.{Threading, BinarySerializer}
import cc.factorie.util.Threading._
import scala.Some
import java.util.concurrent.ExecutorService
import akka.actor.{Actor, Props, ActorSystem}
import concurrent.duration.Duration
import concurrent.{Future, Await}
import concurrent.ExecutionContext.Implicits.global
import akka.actor.Status.Success
import java.io._
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import scala.Some
import akka.actor.Status.Success
import scala.Some
import akka.actor.Status.Success
import scala.Some
import akka.actor.Status.Success
import scala.Some
import akka.actor.Status.Success




class MatrixFactorizationOptions  extends cc.factorie.util.DefaultCmdOptions{
  val numWords = new CmdOption("num-words", "", "INT", "Number of Words")
  val latentDimensionality = new CmdOption("latent-dim", "", "INT", "latent dimensionality")
  val trainFile = new CmdOption("train", "", "FILE", "where to get training triples from")
  val outFile = new CmdOption("out","","FILE","where to write out the learned matrix")
  val numPasses = new CmdOption("num-passes","1","INT","how many passes over the dataset")
  val numExamples = new CmdOption("num-examples","-1","INT","how many examples to use")
}

object MatrixFactorization{
  val opts = new MatrixFactorizationOptions
  val random = new Random(0)

  def main(args: Array[String]): Unit = {
    opts.parse(args)


    //////////////initialize model
    val numRows = opts.numWords.value.toInt
    val numCols = opts.numWords.value.toInt
    val latentDimensionality = opts.latentDimensionality.value.toInt
    val scale = 0.1
    def initVector(i: Int): Array[Double] = Array.fill[Double](latentDimensionality)(scale*random.nextGaussian())

    val rowRegularizer = 0.05
    val colRegularizer = 0.05
    println("allocating model")
    assert(opts.outFile.wasInvoked)

    val model = new SymmetricMatrixFactorizationModel(latentDimensionality,numRows,rowRegularizer,initVector,OptimizableObjectives.logBinary, OptimizableObjectives.logisticLinkFunction)  with CellScoreBinary
    //////////////

    val examplesFull: Iterator[(Int,Int,Int,Int)] = (0 until opts.numPasses.value.toInt).toIterator.flatMap( i=> BinaryTriples.readAsciiTriplesFromFileToQuads(opts.trainFile.value))
    val examples = if(!opts.numExamples.wasInvoked) examplesFull else examplesFull.take(opts.numExamples.value.toInt)
    //
    val start = System.currentTimeMillis()
    train(model,examples)
    println("train time " + .001*(System.currentTimeMillis() - start))
    model.serialize(opts.outFile.value)


  }

  def train(model: MatrixFactorizationModel,examples: Iterator[(Int,Int,Int,Int)]){
    var objective = 0.0
    val numNegatives = 3

    val blockSize = 10000
    val batchedExamples = examples.grouped(blockSize)
    object objectiveMutex

      val stepsize = 0.1//0.1/math.sqrt(t+1)
      var iter = 0
    var totalProcessed = 0
      ProducerConsumerProcessing.parForeach(batchedExamples,Runtime.getRuntime.availableProcessors())(exs => {
        val thisObj =  updateFromCells(model,exs,stepsize,numNegatives)

        objectiveMutex.synchronized{
          objective += thisObj; iter+=1;
          totalProcessed += exs.length


          if(iter % 100 == 0){
            println("finished  " + iter*blockSize + " objective = " + objective/totalProcessed)
            objective = 0.0
            totalProcessed = 0
          }
        }

      })

  }

  def updateFromCells(model: MatrixFactorizationModel, cells: Seq[(Int,Int,Int,Int)], stepsize: Double, numNegatives: Int) : Double = {
    var obj  = 0.0
    cells.foreach(cell => {
        obj += model.updateCell(cell,true,stepsize)
        (0 until numNegatives).foreach(_ => {
          val negCell = (cell._1,random.nextInt(model.numCols),0,0)
          obj += model.updateCell(negCell,false,stepsize)
          val negCell2 = (random.nextInt(model.numRows),cell._2,0,0)
          obj += model.updateCell(negCell2,false,stepsize)
        })
    })
    obj
  }


}


object GetCounts{
  def main(args: Array[String]): Unit = {
    val tripleCounts = collection.mutable.HashMap[(Int,Int,Int),Int]().withDefaultValue(0)
    val pairCounts = collection.mutable.HashMap[(Int,Int),Int]().withDefaultValue(0)

    val iter = BinaryTriples.readAsciiTriplesFromFile(args(0))
    iter.foreach(triple => {
       val pair = (triple._1,triple._2)
       pairCounts(pair) += 1
       tripleCounts(triple) += 1
    })
    println("read triples")
    val writer1 = new PrintWriter(new OutputStreamWriter(
      new GZIPOutputStream(new FileOutputStream(args(1) + ".triple")), "UTF-8"))
    tripleCounts.iterator.foreach(it => {
       writer1.println("%d %d %d".format(it._1._1,it._1._2,it._1._3) + " " + it._2)
    })
    writer1.flush(); writer1.close()

    println("wrote triple counts")

    val writer2 = new PrintWriter(new OutputStreamWriter(
      new GZIPOutputStream(new FileOutputStream(args(1) + ".pair")), "UTF-8"))
    pairCounts.iterator.foreach(it => {
      val str = "%d %d %d".format(it._1._1,it._1._2,it._2)
      writer2.println(str+ " " + it._2)
    })
    println("wrote pair counts")

    writer2.flush(); writer2.close()
  }
}


 //todo: make stuff use counts

object BinaryQuadruples{

  def readAsciiQuadruplesFromFile(f: String,gzip: Boolean = true): Iterator[(Int,Int,Int,Int)] = {
    val reader =
      if (!gzip)
        io.Source.fromFile(f)
      else
        io.Source.fromInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))))

    reader.getLines.map(line => {
      val fields = line.split(" ")
      (fields(0).toInt , fields(1).toInt, fields(2).toInt,fields(3).toInt)
    })
  }
}

object BinaryTriples{
//  def main(args: Array[String]): Unit = {
//      val iter = readAsciiTriplesFromFile(args(0))
//      serialize(iter,args(1))
//  }
//  def serialize(iter: Iterator[(Int,Int,Int)], out: String) : Unit = {
//    val writer = new DataOutputStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(out))))
//    iter.foreach(abc => {writer.writeInt(abc._1); writer.writeInt(abc._2); writer.writeInt(abc._3)} )
//    writer.flush()
//    writer.close()
//  }
//  def deserialize(in: String): Iterator[(Int,Int,Int)] = {
//    val reader = new DataInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(in))))
//
//    var nextTriple = null: (Int,Int,Int)
//    getNext()
//    def getNext(): Unit  = {
//      try {
//        nextTriple = (reader.readInt(),reader.readInt(),reader.readInt())
//      } catch{
//        case e: EOFException => nextTriple = null
//      }
//    }
//
//    new Iterator[(Int,Int,Int)]{
//      def hasNext = nextTriple != null
//      def next() = {
//        val d = nextTriple
//        getNext()
//        d
//      }
//    }
//
//  }
  def readAsciiTriplesFromFileToQuads(f: String,gzip: Boolean = true): Iterator[(Int,Int,Int,Int)] = {
    val reader =
    if (!gzip)
      io.Source.fromFile(f)
    else
      io.Source.fromInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))))

    reader.getLines.map(line => {
      val fields = line.split(" ")
      (fields(0).toInt , fields(1).toInt, fields(2).toInt,0)
    })
  }
  def readAsciiTriplesFromFile(f: String,gzip: Boolean = true): Iterator[(Int,Int,Int)] = {
    val reader =
      if (!gzip)
        io.Source.fromFile(f)
      else
        io.Source.fromInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))))

    reader.getLines.map(line => {
      val fields = line.split(" ")
      (fields(0).toInt , fields(1).toInt, fields(2).toInt)
    })
  }
}


object ProducerConsumerProcessing{
  import akka.pattern.ask

  object IteratorMutex
  def parForeach[In](xs: Iterator[In], numParallelJobs: Int = Runtime.getRuntime.availableProcessors(),perJobTimeout: Long = 10 ,overallTimeout: Long = 24)(body: In => Unit): Unit  = {
    val system = ActorSystem("producer-consumer")

    val actors = (0 until numParallelJobs).map(i => system.actorOf(Props(new ParForeachActor(body)), "actor-"+i))


    val futures = actors.map(a => a.ask(Message(xs))(Duration(overallTimeout,"hours")))
    Await.result(Future.sequence(futures.toSeq), Duration(overallTimeout,"hours"))
    system.shutdown()

  }

  class ParForeachActor[In](function: In => Unit) extends Actor {
    def getNext(a: Iterator[In]): Option[In] = {
      IteratorMutex synchronized {
        if (a.hasNext)
          Some(a.next())
        else
          None
      }
    }
    def receive = {
      case Message(a) => {
        var stillWorking = true
        while (stillWorking) {
          val next = getNext(a.asInstanceOf[Iterator[In]])
          if(next.isDefined) function(next.get) else stillWorking = false
        }
        sender ! Success
      }
    }
  }
  case class Message[T](a: Iterator[T])

}