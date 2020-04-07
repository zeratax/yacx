import java.util.Arrays;

import yacx.FloatArg;
import yacx.HalfArg;

public class MatrixUtils {
	//TODO Die Methoden sind total langsam
	public static void printlnMatrix(FloatArg floatMatrix, int columns) {
		int rows = floatMatrix.getLength()/columns;
		
		for (int r = 0; r < rows-1; r++) {
			System.out.println(Arrays.toString(floatMatrix.slice(r*columns, (r+1)*columns-1).asFloatArray()));
		}
		System.out.println(Arrays.toString(floatMatrix.slice((rows-1)*columns, rows*columns-1).asFloatArray()));
	}
	
	public static void printlnMatrix(HalfArg halfMatrix, int columns) {
		int rows = halfMatrix.getLength()/columns;
		
		for (int r = 0; r < rows-1; r++) {
			System.out.println(Arrays.toString(halfMatrix.slice(r*columns, (r+1)*columns-1).asFloatArray()));
		}
		System.out.println(Arrays.toString(halfMatrix.slice((rows-1)*columns, rows*columns-1).asFloatArray()));
	}
}
