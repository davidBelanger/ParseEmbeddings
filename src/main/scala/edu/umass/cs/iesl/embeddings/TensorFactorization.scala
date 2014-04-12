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


trait FactorizationModel{
  def updateCell(cell: (Int,Int,Int),target: Boolean,stepsize: Double): Double
  def serialize(outFile: String): Unit
}

class AsymmetricMatrixFactorizationModel(latentDim: Int,val numRows: Int,val numCols: Int,rowRegularizer: Double, colRegularizer: Double,initializeRow: Int => Array[Double],initializeCol: Int => Array[Double],objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends MatrixFactorizationModel(latentDim,rowRegularizer, colRegularizer,objective,linkFunction){
  private val rowVectors = (0 until numRows).map(i => new DenseTensor1(initializeRow(i)))
  private val colVectors = (0 until numCols).map(i => new DenseTensor1(initializeCol(i)))
  def rowVector(i: Int): DenseTensor1  = rowVectors(i)
  def colVector(j: Int): DenseTensor1 = colVectors(j)
  def serialize(out: String) = {
    EmbeddingSerialization.serialize(out + ".rows",rowVectors)
    EmbeddingSerialization.serialize(out + ".cols",colVectors)
  }
}

class SymmetricMatrixFactorizationModel(latentDim: Int,numRowsCols: Int,regularizer: Double,initialize: Int => Array[Double],objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends MatrixFactorizationModel(latentDim,regularizer, regularizer,objective,linkFunction){
  private val vectors = (0 until numRowsCols).map(i => new DenseTensor1(initialize(i)))
  def rowVector(i: Int): DenseTensor1  = vectors(i)
  def colVector(j: Int): DenseTensor1 = vectors(j)
  val numRows = numRowsCols
  val numCols = numRowsCols
  def serialize(out: String) = {
    EmbeddingSerialization.serialize(out,vectors)
  }
}

abstract class MatrixFactorizationModel(val latentDim: Int,val rowRegularizer: Double, val colRegularizer: Double,val objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends FactorizationModel{

  def rowVector(i: Int): DenseTensor1
  def colVector(j: Int): DenseTensor1
  def numRows: Int
  def numCols: Int

  def lossAndGradient(i: Int,j:Int,cellIsTrue: Boolean) = objective.valueAndGradient(score(i,j),if(cellIsTrue) 1 else -1)
  def predict(i: Int,j: Int) = linkFunction(score(i,j))
  def score(i: Int, j: Int) = rowVector(i).dot(colVector(j))
  def updateCell(cell: (Int,Int,Int),target: Boolean,stepsize: Double): Double = {
    val rowIndex = cell._1
    val colIndex = cell._2
    val lg = lossAndGradient(rowIndex,colIndex,target)
    val thisObjective = lg._1
    val step = stepsize*lg._2
    rowVector(rowIndex).*=((1-stepsize*rowRegularizer))
    rowVector(rowIndex).+=(colVector(colIndex),step)

    colVector(colIndex).*=((1-stepsize*colRegularizer))
    colVector(colIndex).+=(rowVector(rowIndex),step)

    thisObjective
  }

}


class MatrixFactorizationOptions  extends cc.factorie.util.DefaultCmdOptions{
  val numWords = new CmdOption("num-words", "", "INT", "Number of Words")
  val latentDimensionality = new CmdOption("latent-dim", "", "INT", "latent dimensionality")
  val trainFile = new CmdOption("train", "", "FILE", "where to get training triples from")
  val outFile = new CmdOption("out","","FILE","where to write out the learned matrix")
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
    val model = new SymmetricMatrixFactorizationModel(latentDimensionality,numRows,rowRegularizer,initVector,OptimizableObjectives.logBinary, OptimizableObjectives.logisticLinkFunction)
    //////////////

    val examples: Iterator[(Int,Int,Int)] = readTriplesFromFile(opts.trainFile.value)
    train(model,examples)
    model.serialize(opts.outFile.value)


  }

  def train(model: MatrixFactorizationModel,examples: Iterator[(Int,Int,Int)]){
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


          if(iter % 25 == 0){
            println("finished  " + iter*blockSize + " objective = " + objective/totalProcessed)
            objective = 0.0
            totalProcessed = 0
          }
        }

      })

  }

  def updateFromCells(model: MatrixFactorizationModel, cells: Seq[(Int,Int,Int)], stepsize: Double, numNegatives: Int) : Double = {
    var obj  = 0.0
    cells.foreach(cell => {
        obj += model.updateCell(cell,true,stepsize)
        (0 until numNegatives).foreach(_ => {
          val negCell = (cell._1,random.nextInt(model.numCols),0)
          obj += model.updateCell(negCell,false,stepsize)
          val negCell2 = (random.nextInt(model.numRows),cell._2,0)
          obj += model.updateCell(negCell2,false,stepsize)
        })
    })
    obj
  }

  def readTriplesFromFile(f: String): Iterator[(Int,Int,Int)] = {
    io.Source.fromFile(f).getLines.map(line => {
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