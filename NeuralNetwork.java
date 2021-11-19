
public class NeuralNetwork {
	
	private double learningRate = 0.1;
	
	private int inputNodes;
	private int outputNodes;
	private int hiddenNodes;
	private Matrix weightsIH;
	private Matrix weightsHO;
	private Matrix biasH;
	private Matrix biasO;
	
	public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
		this.inputNodes = inputNodes;
		this.hiddenNodes = hiddenNodes;
		this.outputNodes = outputNodes;
		
		weightsIH = Matrix.random(hiddenNodes, inputNodes);
		weightsHO = Matrix.random(outputNodes, hiddenNodes);
		
		biasH = Matrix.random(hiddenNodes, 1);
		biasO = Matrix.random(outputNodes, 1);
		
	}
	
	public double[] feedforward(double[] inputArray) {
		double[][] inputs2D = new double[inputArray.length][1];
		for(int i = 0; i < inputArray.length; i++) {
			inputs2D[i][0] = inputArray[i];
		}
		Matrix inputs = new Matrix(inputs2D);
		Matrix hidden = weightsIH.times(inputs);
		hidden = hidden.plus(biasH);
		hidden = hidden.sigmoid();
		
		Matrix output = weightsHO.times(hidden);
		output = output.plus(biasO);
		output = output.sigmoid();
		
		double[][] values = output.getValues();
		double[] out = new double[outputNodes];
		int count = 0;
		for(int i = 0; i < values.length; i++) {
			for(int j = 0; j < values[0].length; j++) {
				out[count] = values[i][j];
				count++;
			}
		}
		return out;
	}
	
	public void train(double[] inputArray, double[] targetArray) {
		double[][] inputs2D = new double[inputArray.length][1];
		double[][] targets2D = new double[targetArray.length][1];
		for(int i = 0; i < inputArray.length; i++) {
			inputs2D[i][0] = inputArray[i];
		}
		for(int i = 0; i < targetArray.length; i++) {
			targets2D[i][0] = targetArray[i];
		}
		Matrix inputs = new Matrix(inputs2D);
		Matrix hidden = weightsIH.times(inputs);
		hidden = hidden.plus(biasH);
		hidden = hidden.sigmoid();
		
		Matrix output = weightsHO.times(hidden);
		output = output.plus(biasO);
		output = output.sigmoid();
		
		Matrix targets = new Matrix(targets2D);
		
		Matrix outputErrors = targets.minus(output);
		
		Matrix gradients = output.dsigmoid();
		gradients = gradients.times(outputErrors);
		gradients = gradients.times(learningRate);
		
		Matrix hiddenT = hidden.transpose();
		Matrix weightHODeltas = gradients.times(hiddenT);
		
		weightsHO = weightsHO.plus(weightHODeltas);
		biasO = biasO.plus(gradients);
		
		Matrix whoT = weightsHO.transpose();
		Matrix hiddenErrors = whoT.times(outputErrors);
		
		Matrix hiddenGradient = hidden.dsigmoid();
		hiddenGradient = hiddenGradient.hadamard(hiddenErrors);
		hiddenGradient = hiddenGradient.times(learningRate);
		
		Matrix inputsT = inputs.transpose();
		Matrix weightIHDeltas = hiddenGradient.times(inputsT);
		
		weightsIH = weightsIH.plus(weightIHDeltas);
		biasH = biasH.plus(hiddenGradient);
	}
	
	public void setLearningRate(double lr) {
		learningRate = lr;
	}
}
