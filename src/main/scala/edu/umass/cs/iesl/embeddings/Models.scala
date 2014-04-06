package edu.umass.cs.iesl.embeddings

import cc.factorie.la.{WeightsMapAccumulator, DenseTensor1, DenseTensor2, Tensor1}
import cc.factorie.model.Parameters
import cc.factorie.app.classify.backend.OptimizablePredictor


//this assumes that the dense vector is of length the number of classes and that class's weight vector for multiclass classification just appends this one-dimensional dense feature
class SparseAndDenseClassConditionalLinearMulticlassClassifier[T1 <: Tensor1, T2 <: Tensor1](labelSize: Int, sparseFeatureSize: Int) extends cc.factorie.app.classify.backend.MulticlassClassifier[(T1,T2)] with Parameters with OptimizablePredictor[Tensor1,(T1,T2)] {
  val weightsForSparseFeatures = Weights(new DenseTensor2(sparseFeatureSize, labelSize))
  val weightsForDenseFeatures =  (0 until labelSize).map(i => Weights(new DenseTensor1(1)))

  def predict(features: (T1,T2)): Tensor1 = {
    val result = weightsForSparseFeatures.value.leftMultiply(features._1)
    (0 until labelSize).foreach(i => {
      val term = weightsForDenseFeatures(i).value(0) * features._2(i)
      result.+=(i,term)
    })
    result
  }
  def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, features: (T1,T2), gradient: Tensor1, weight: Double) = {
    accumulator.accumulate(weightsForSparseFeatures, features._1 outer gradient)
    (0 until labelSize).foreach(i => {
      accumulator.accumulate(weightsForDenseFeatures(i), new DenseTensor1(1,features._2(i) * gradient(i)))
    })
  }
}

class SparseAndDenseLinearMulticlassClassifier[T1 <: Tensor1, T2 <: Tensor1](labelSize: Int, sparseFeatureSize: Int, denseFeatureSize: Int) extends cc.factorie.app.classify.backend.MulticlassClassifier[(T1,T2)] with Parameters with OptimizablePredictor[Tensor1,(T1,T2)] {
  val weightsForSparseFeatures = Weights(new DenseTensor2(sparseFeatureSize, labelSize))
  val weightsForDenseFeatures = Weights(new DenseTensor2(denseFeatureSize, labelSize))

  def predict(features: (T1,T2)): Tensor1 = {
    val result = weightsForSparseFeatures.value.leftMultiply(features._1)
    result.+=(weightsForDenseFeatures.value.leftMultiply(features._2))
    result
  }
  def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, features: (T1,T2), gradient: Tensor1, weight: Double) = {
    accumulator.accumulate(weightsForSparseFeatures, features._1 outer gradient)
    accumulator.accumulate(weightsForDenseFeatures, features._2  outer gradient)
  }
}
