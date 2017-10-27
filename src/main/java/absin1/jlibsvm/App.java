package absin1.jlibsvm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.SvmProblem;
import edu.berkeley.compbio.jlibsvm.binary.BinaryModel;
import edu.berkeley.compbio.jlibsvm.binary.C_SVC;
import edu.berkeley.compbio.jlibsvm.binary.MutableBinaryClassificationProblemImpl;
import edu.berkeley.compbio.jlibsvm.kernel.LinearKernel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * Hello world!
 *
 */
public class App {
	public static void main(String[] args) throws IOException {
		App app = new App();
		// BinaryModel trainSVM = app.trainSVM();
		// app.testSVM(trainSVM);
		app.trainandTest();
	}

	private void trainandTest() throws IOException {
		File file = new File("/home/ab/Downloads/rcv1_train.binary");
		FileReader fileReader = new FileReader(file);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		String line = null;
		int dataEntrySize = 0;
		while ((line = bufferedReader.readLine()) != null) {
			dataEntrySize++;
		}
		System.err.println(dataEntrySize);
		MutableBinaryClassificationProblemImpl problem = new MutableBinaryClassificationProblemImpl(String.class,
				dataEntrySize);
		bufferedReader = new BufferedReader(fileReader);
		while ((line = bufferedReader.readLine()) != null) {
			String[] linesplit = line.split(" ");
			int[] indices = new int[linesplit.length - 1];
			float[] floats = new float[linesplit.length - 1];
			Comparable label = linesplit[0];
			for (int i = 1; i < linesplit.length; i++) {
				String[] indexTFIDFsplit = linesplit[i].split(":");
				indices[i - 1] = new Integer(indexTFIDFsplit[0]);
				floats[i - 1] = new Float(Double.parseDouble(indexTFIDFsplit[1]));
				SparseVector sparseVector = new SparseVector(indices.length);
				sparseVector.indexes = indices;
				sparseVector.values = floats;
				problem.addExample(sparseVector, label);
			}
		}
		C_SVC svm = new C_SVC();
		ImmutableSvmParameterGrid.Builder builder = ImmutableSvmParameterGrid.builder();
		HashSet<Float> cSet;
		HashSet<LinearKernel> kernelSet;

		cSet = new HashSet<Float>();
		cSet.add(1.0f);

		kernelSet = new HashSet<LinearKernel>();
		kernelSet.add(new LinearKernel());

		// configure finetuning parameters
		builder.eps = 0.001f; // epsilon
		builder.Cset = cSet; // C values used
		builder.kernelSet = kernelSet; // Kernel used
		ImmutableSvmParameter params = builder.build();
		BinaryModel<Comparable, SparseVector> model = svm.train(problem, params);

		file = new File("/home/ab/Downloads/rcv1_test.binary");
		fileReader = new FileReader(file);
		bufferedReader = new BufferedReader(fileReader);
		line = null;
		while ((line = bufferedReader.readLine()) != null) {
			String[] linesplit = line.split(" ");
			int[] indices = new int[linesplit.length - 1];
			float[] floats = new float[linesplit.length - 1];
			String actualLabel = linesplit[0];
			SparseVector XsparseVector = new SparseVector(linesplit.length - 1);
			for (int i = 1; i < linesplit.length; i++) {
				String[] indexTFIDFsplit = linesplit[i].split(":");
				indices[i - 1] = Integer.parseInt(indexTFIDFsplit[0]);
				floats[i - 1] = new Float(Double.parseDouble(indexTFIDFsplit[1]));
				XsparseVector.indexes = indices;
				XsparseVector.values = floats;
			}
			try {
				String predictValue = model.predictValue(XsparseVector).toString();
				System.out.println("predicted:" + predictValue);
			} catch (NullPointerException e) {
				e.printStackTrace();
			}
			System.err.println(" against actual:" + actualLabel);
		}
	}

	private void testSVM(BinaryModel model) throws NumberFormatException, IOException {
		File file = new File("/home/ab/Downloads/rcv1_test.binary");
		FileReader fileReader = new FileReader(file);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		String line = null;
		while ((line = bufferedReader.readLine()) != null) {
			String[] linesplit = line.split(" ");
			int[] indices = new int[linesplit.length - 1];
			float[] floats = new float[linesplit.length - 1];
			String actualLabel = linesplit[0];
			SparseVector sparseVector = new SparseVector(indices.length);
			for (int i = 1; i < linesplit.length; i++) {
				String[] indexTFIDFsplit = linesplit[i].split(":");
				indices[i - 1] = Integer.parseInt(indexTFIDFsplit[0]);
				floats[i - 1] = new Float(Double.parseDouble(indexTFIDFsplit[1]));
				sparseVector.indexes = indices;
				sparseVector.values = floats;
			}
			try {
				int predictedLabel = (Integer) model.predictLabel(sparseVector);
				System.out.println("predicted:" + predictedLabel + " against actual:" + actualLabel);
			} catch (NullPointerException e) {
				e.printStackTrace();
			}
		}
	}

	private BinaryModel trainSVM() throws FileNotFoundException, IOException {
		File file = new File("/home/ab/Downloads/rcv1_train.binary");
		FileReader fileReader = new FileReader(file);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		String line = null;
		int dataEntrySize = 0;
		while ((line = bufferedReader.readLine()) != null) {
			dataEntrySize++;
		}
		System.err.println(dataEntrySize);
		MutableBinaryClassificationProblemImpl problem = new MutableBinaryClassificationProblemImpl(Float.class, 2);
		bufferedReader = new BufferedReader(fileReader);
		while ((line = bufferedReader.readLine()) != null) {
			String[] linesplit = line.split(" ");
			int[] indices = new int[linesplit.length - 1];
			float[] floats = new float[linesplit.length - 1];
			Comparable label = linesplit[0];
			for (int i = 1; i < linesplit.length; i++) {
				String[] indexTFIDFsplit = linesplit[i].split(":");
				indices[i - 1] = new Integer(indexTFIDFsplit[0]);
				floats[i - 1] = new Float(Double.parseDouble(indexTFIDFsplit[1]));
				SparseVector sparseVector = new SparseVector(indices.length);
				sparseVector.indexes = indices;
				sparseVector.values = floats;
				problem.addExample(sparseVector, label);
			}
		}
		C_SVC svm = new C_SVC();
		ImmutableSvmParameterGrid.Builder builder = ImmutableSvmParameterGrid.builder();
		HashSet<Float> cSet;
		HashSet<LinearKernel> kernelSet;

		cSet = new HashSet<Float>();
		cSet.add(1.0f);

		kernelSet = new HashSet<LinearKernel>();
		kernelSet.add(new LinearKernel());

		// configure finetuning parameters
		builder.eps = 0.001f; // epsilon
		builder.Cset = cSet; // C values used
		builder.kernelSet = kernelSet; // Kernel used
		ImmutableSvmParameter params = builder.build();
		BinaryModel model = svm.train(problem, params);
		return model;
	}

}
