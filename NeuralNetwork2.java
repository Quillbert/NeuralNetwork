public class NeuralNetwork2 {
	public double learningRate = 0.001;
	private int[] nodeCounts; //size is layers; number of each layer is number of nodes per layer
	private Matrix[] weights;//weights of each node of a layer to each node of the next layer
	private Matrix[] biases;//increases the intensity
	public NeuralNetwork2(int... nodeCounts) {// allows any number of layers to be included
		this.nodeCounts = nodeCounts;
		weights = new Matrix[nodeCounts.length-1];
		biases = new Matrix[nodeCounts.length-1];
		for(int i = 0; i < weights.length; i++) { //initialize weights and biases with random values
			weights[i] = Matrix.random(nodeCounts[i+1], nodeCounts[i]);
			biases[i] = Matrix.random(nodeCounts[i+1], 1);
		}
	}
	public double[] feedforward(double[] inputArray) {
		double[][] inputs2D = new double[inputArray.length][1];//changes inputs(starting array) into a 2D array
		for(int i = 0; i < inputArray.length; i++) {
			inputs2D[i][0] = inputArray[i];
		}
		Matrix inputs = new Matrix(inputs2D);//changes to a matrix
		Matrix output = null;
		for(int i = 0; i < nodeCounts.length-1; i++) {//through every layer
			output = weights[i].times(inputs);//output(next layer) is Sigmoid(W1*V1 + B1)
			output = output.plus(biases[i]);
			output = output.sigmoid();
			inputs = output;//increments to next layer
		}
		double[][] values = output.getValues();//converting output to 1D array
		double[] out = new double[nodeCounts[nodeCounts.length-1]];//out length is the value of the number of output choices
		int count = 0;
		for(int i = 0; i < values.length; i++) {
			for(int j = 0; j < values[0].length; j++) {
				out[count] = values[i][j];//converts output to a 1D array
				count++;
			}
		}
		return out;//outputs 1D array
	}
	public void train(double[] inputArray, double[] targetArray) {
		double[][] inputs2D = new double[inputArray.length][1];//inputs and known values
		double[][] targets2D = new double[targetArray.length][1];
		for(int i = 0; i < inputArray.length; i++) {//changes each to 1D
			inputs2D[i][0] = inputArray[i];
		}
		for(int i = 0; i < targetArray.length; i++) {
			targets2D[i][0] = targetArray[i];
		}
		Matrix[] layers = new Matrix[nodeCounts.length];//array of matrices of values for each layer
		layers[0] = new Matrix(inputs2D);//first layer is inputs
		for(int i = 1; i < nodeCounts.length; i++) {//walks through number of layers
			layers[i] = weights[i-1].times(layers[i-1]);//each layer is Sigmoid(Wi-1 * Vi-1 + Bi-1); Same as feed forward
			layers[i] = layers[i].plus(biases[i-1]);
			layers[i] = layers[i].sigmoid();
		}
		Matrix targets = new Matrix(targets2D);//converts correct answers into matrix
		Matrix errors = targets.minus(layers[layers.length-1]);//gets matrix of errors for each attempt
		//errors = errors.hadamard(errors);//square the value of the errors
		for(int i = nodeCounts.length-1; i > 0 ; i--) {//traverses layers backwards
			//finds changes for greatest alteration in favor of the proper answer for each weight
			Matrix gradients = layers[i].dsigmoid();//gradients is (Sigmoid'(V)+error) * learningRate
			gradients = gradients.hadamard(errors);
			gradients = gradients.times(learningRate);
			Matrix valuesTransposed = layers[i-1].transpose();//transposed previous layer is multiplied to gradient
			Matrix weightDeltas = gradients.times(valuesTransposed);//results in necessary changes to weights
			weights[i-1] = weights[i-1].plus(weightDeltas);//weights change based off of input values
			biases[i-1] = biases[i-1].plus(gradients);//biases changes based off of shift of values needed
			Matrix weightsTransposed = weights[i-1].transpose();//transposed weights of previous section
			errors = weightsTransposed.times(errors);//multiply by errors to get errors of previous section
		}
	}
}