import java.io.IOException;
import java.util.*;

/**
 * 
 * @author Andrew Hamby and Nolan Pozzobon
 * Honor Pledge: All work here is honestly obtained and is my own.
 * Assignment: Final Project: Neural Network Example
 * Resources:
 		http://neuralnetworksanddeeplearning.com/chap1.html
		https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
		https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh
		Artificial Intelligence: A Modern Approach - Chapter 18
		https://introcs.cs.princeton.edu/java/95linear/Matrix.java.html
		https://github.com/turkdogan/mnist-data-reader
		http://yann.lecun.com/exdb/mnist/

 *
 */

public class Tester {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		//Hard Coded 3 Layer Test
		/*NeuralNetwork2 network = new NeuralNetwork2(2, 3, 1);
		double[][] inputs = {{1, 1}, {0, 0}, {0, 1}, {1, 0}};
		double[][] targets = {{0}, {0}, {1}, {1}};
		for(int i = 0; i < 100000; i++) {
			int index = (int)(Math.random()*4);
			network.train(inputs[index], targets[index]);
		}
		for(int i = 0; i < 4; i++) {
			double[] outputs = network.feedforward(inputs[i]);
			System.out.println(Arrays.toString(outputs));
		}*/

		NeuralNetwork2 network = new NeuralNetwork2(784, 30, 10);//number of layers
		//training set, small set of data
		MnistMatrix[] trainingSet = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
		System.out.println("Training Set Loaded");
		//test set
		MnistMatrix[] testSet = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
		System.out.println("Testing Set Loaded");
		for(int times = 0; times < 500; times++) {//500 times through the training
			LinkedList<MnistMatrix> data = new LinkedList<MnistMatrix>(Arrays.asList(trainingSet));//list of test items
			for(int i = 0; i < trainingSet.length; i++) {
				MnistMatrix test = data.remove((int)(Math.random()*data.size()));//each test is removed from list
				double[] inputs = new double[784];//28x28 pixles
				double[] targets = new double[10];//10 possible outputs
				int counter = 0;
				for(int j = 0; j < 28; j++) {
					for(int k = 0; k < 28; k++) {
						inputs[counter] = test.getValue(j, k);//data is changed into 1D array
						counter++;
					}
				}
				targets[test.getLabel()] = 1;
				network.train(inputs, targets);//trains based off of test set
			}
			int correct = 0;
			System.out.println("Accuracy: " + evaluateAccuracy(network, testSet)*100 + "%");
			//network.learningRate *= 0.999;//increments down learning rate to be more specific
		}
		System.out.println("Training Complete");
		//checks on test set
		int correct = 0;
		for(int i = 0; i < testSet.length; i++) {
			double[] inputs = new double[784];
			int counter = 0;
			for(int j = 0; j < 28; j++) {
				for(int k = 0; k < 28; k++) {
					inputs[counter] = testSet[i].getValue(j, k);
					counter++;
				}
			}
			double[] output = network.feedforward(inputs);
			if(guess(output) == testSet[i].getLabel()) {
				correct++;
			}
		}
		System.out.println("Final Accuracy: " + ((double)correct/testSet.length)*100 + "%");
	}

	//returns the node from the final section with the greatest activation
	public static int guess(double[] outputs) {
		int guess = 0;
		double max = outputs[0];
		for(int i = 1; i < outputs.length; i++) {
			if(outputs[i] > max) {
				guess = i;
				max = outputs[i];
			}
		}
		return guess;
	}
	private static void printMnistMatrix(final MnistMatrix matrix) {
		System.out.println("label: " + matrix.getLabel());
		for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
			for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
				System.out.print(matrix.getValue(r, c) + " ");
			}
			System.out.println();
		}
	}
	
	private static double evaluateAccuracy(NeuralNetwork2 network, MnistMatrix[] testSet) {
		int correct = 0;
		for(int i = 0; i < testSet.length; i++) {//goes through the test set
			double[] inputs = new double[784];
			int counter = 0;
			for(int j = 0; j < 28; j++) {
				for(int k = 0; k < 28; k++) {
					inputs[counter] = testSet[i].getValue(j, k);//converts test set to a 1Darray
					counter++;
				}
			}
			double[] output = network.feedforward(inputs);
			if(guess(output) == testSet[i].getLabel()) {//checks guess on test item
				correct++;//number of correct items increases
			}
		}
		return (double)correct/testSet.length;
	}

}