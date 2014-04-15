package edu.umass.cs.iesl.embeddings

import cc.factorie.optimize.{OptimizableObjectives, UnivariateOptimizableObjective}
import cc.factorie.la.DenseTensor1


///General Stuff
trait FactorizationModel{
  def updateCell(cell: (Int,Int,Int,Int),target: Boolean,stepsize: Double): Double
  def serialize(outFile: String): Unit
}

trait CellScoreBinary{
  def cellWiseLossAndGradient(cellScore: Double,cell: (Int,Int,Int,Int),observedSample: Boolean,objective: UnivariateOptimizableObjective[Int]): (Double,Double) = {
    if(observedSample)
      objective.valueAndGradient(cellScore,1)
    else
      objective.valueAndGradient(cellScore,-1)
  }
}
trait CellScoreCounts{
  def cellWiseLossAndGradient(cellScore: Double,cell: (Int,Int,Int,Int),observedSample: Boolean,objective: UnivariateOptimizableObjective[Int]): (Double,Double) = {
    if(observedSample)
      objective.valueAndGradient(cellScore,cell._4)
    else
      objective.valueAndGradient(cellScore,0)
  }
}

///Matrix Stuff

abstract class AsymmetricMatrixFactorizationModel(latentDim: Int,val numRows: Int,val numCols: Int,rowRegularizer: Double, colRegularizer: Double,initializeRow: Int => Array[Double],initializeCol: Int => Array[Double],objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends MatrixFactorizationModel(latentDim,rowRegularizer, colRegularizer,objective,linkFunction){
  private val rowVectors = (0 until numRows).map(i => new DenseTensor1(initializeRow(i)))
  private val colVectors = (0 until numCols).map(i => new DenseTensor1(initializeCol(i)))
  def rowVector(i: Int): DenseTensor1  = rowVectors(i)
  def colVector(j: Int): DenseTensor1 = colVectors(j)
  def serialize(out: String) = {
    EmbeddingSerialization.serialize(out + ".rows",rowVectors)
    EmbeddingSerialization.serialize(out + ".cols",colVectors)
  }
}

abstract class SymmetricMatrixFactorizationModel(latentDim: Int,numRowsCols: Int, regularizer: Double, initialize: Int => Array[Double],objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends MatrixFactorizationModel(latentDim,regularizer, regularizer,objective,linkFunction){
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
  def cellWiseLossAndGradient(cellScore: Double,cell: (Int,Int,Int,Int),observedSample: Boolean,objective: UnivariateOptimizableObjective[Int]): (Double,Double)

  def predict(i: Int,j: Int) = linkFunction(score(i,j))
  def score(i: Int, j: Int) = rowVector(i).dot(colVector(j))
  def updateCell(cell: (Int,Int,Int,Int),observedSample: Boolean,stepsize: Double): Double = {
    val rowIndex = cell._1
    val colIndex = cell._2
    val cellScore = score(rowIndex,colIndex )
    val lg = cellWiseLossAndGradient(cellScore,cell,observedSample,objective)
    val thisObjective = lg._1
    val step = stepsize*lg._2
    rowVector(rowIndex).*=((1-stepsize*rowRegularizer))
    rowVector(rowIndex).+=(colVector(colIndex),step)

    colVector(colIndex).*=((1-stepsize*colRegularizer))
    colVector(colIndex).+=(rowVector(rowIndex),step)

    thisObjective
  }


}

abstract class AsymmetricTensorFactorizationModel(latentDim: Int,val numRows: Int,val numCols: Int,val numThird: Int,rowRegularizer: Double, colRegularizer: Double,thirdRegularizer: Double,initializeRow: Int => Array[Double],initializeCol: Int => Array[Double],initializeThird: Int => Array[Double],objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends TensorFactorizationModel(latentDim,rowRegularizer, colRegularizer,thirdRegularizer,objective,linkFunction){
  private val rowVectors = (0 until numRows).map(i => new DenseTensor1(initializeRow(i)))
  private val colVectors = (0 until numCols).map(i => new DenseTensor1(initializeCol(i)))
  private val thirdVectors = (0 until numThird).map(i => new DenseTensor1(initializeThird(i)))

  def rowVector(i: Int): DenseTensor1  = rowVectors(i)
  def colVector(j: Int): DenseTensor1 = colVectors(j)
  def thirdVector(k: Int): DenseTensor1 = thirdVectors(k)

  def serialize(out: String) = {
    EmbeddingSerialization.serialize(out + ".rows",rowVectors)
    EmbeddingSerialization.serialize(out + ".cols",colVectors)
    EmbeddingSerialization.serialize(out + ".arcs",thirdVectors)
  }
}

abstract class SymmetricTensorFactorizationModel(latentDim: Int,numRowsCols: Int, val numThird: Int,regularizer: Double, thirdRegularizer: Double,initialize: Int => Array[Double],objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends TensorFactorizationModel(latentDim,regularizer, regularizer,thirdRegularizer,objective,linkFunction){
  private val vectors = (0 until numRowsCols).map(i => new DenseTensor1(initialize(i)))
  private val thirdVectors = (0 until numRowsCols).map(i => new DenseTensor1(initialize(i)))

  def rowVector(i: Int): DenseTensor1  = vectors(i)
  def colVector(j: Int): DenseTensor1 = vectors(j)
  def thirdVector(k: Int): DenseTensor1 = thirdVectors(k)

  val numRows = numRowsCols
  val numCols = numRowsCols
  def serialize(out: String) = {
    EmbeddingSerialization.serialize(out + ".words",vectors)
    EmbeddingSerialization.serialize(out + ".arcs",vectors)
  }
}

////Tensor Stuff
abstract class TensorFactorizationModel(val latentDim: Int,val rowRegularizer: Double, val colRegularizer: Double,val thirdRegularizer: Double,val objective: UnivariateOptimizableObjective[Int],linkFunction: OptimizableObjectives.UnivariateLinkFunction) extends FactorizationModel{

  def rowVector(i: Int): DenseTensor1
  def colVector(j: Int): DenseTensor1
  def thirdVector(k: Int): DenseTensor1
  def numRows: Int
  def numCols: Int
  def numThird: Int
  def cellWiseLossAndGradient(cellScore: Double,cell: (Int,Int,Int,Int),observedSample: Boolean,objective: UnivariateOptimizableObjective[Int]): (Double,Double)

  def predict(i: Int,j: Int,k: Int) = linkFunction(score(i,j,k))
  def score(i: Int, j: Int, k: Int) =  TensorUtil.threeWayDotProd(rowVector(i), colVector(j), thirdVector(k))
  def updateCell(cell: (Int,Int,Int,Int),observedSample: Boolean,stepsize: Double): Double = {
    val rowIndex = cell._1
    val colIndex = cell._2
    val thirdIndex = cell._3
    val cellScore = score(rowIndex,colIndex,thirdIndex)
    val lg = cellWiseLossAndGradient(cellScore,cell,observedSample,objective)
    val thisObjective = lg._1
    val step = stepsize*lg._2
    rowVector(rowIndex).*=((1-stepsize*rowRegularizer))
    val everythingButRow = TensorUtil.zipVectors(colVector(colIndex),thirdVector(thirdIndex))
    rowVector(rowIndex).+=(everythingButRow,step)

    colVector(colIndex).*=((1-stepsize*colRegularizer))
    val everythingButCol = TensorUtil.zipVectors(rowVector(rowIndex),thirdVector(thirdIndex))
    colVector(colIndex).+=(everythingButCol,step)

    thirdVector(colIndex).*=((1-stepsize*thirdRegularizer))
    val everythingButThird = TensorUtil.zipVectors(rowVector(rowIndex),colVector(colIndex))
    thirdVector(colIndex).+=(everythingButThird,step)

    thisObjective
  }


}

