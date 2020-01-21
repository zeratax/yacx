package yacx;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestPaddingArg extends TestJNI {
	static int[] matrix1; //2x2 matrix
	static float[] matrix2; //2x6 matrix
	static int columns1, columns2, rows1, rows2;
	static IntArg matrix1Arg;
	static HalfArg matrix2Arg;
	
	static final String copyIntArrayString = "extern \"C\" __global__\n" + 
			"void copyInt(int* in, int* out) {\n" + 
			"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
			"  out[i] = in[i];\n" + 
			"}\n" + 
			"";
	
	static final String copyHalfArrayString = "#include <cuda_fp16.h>\n" +
			"\"extern \"C\" __global__\n" + 
			"void copyHalf(half* in, half* out) {\n" + 
			"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
			"  out[i] = in[i];\n" + 
			"}\n" + 
			"";
	
	@BeforeAll
	static void init() {
		matrix1 = new int[] {1,2,3,4};
		matrix2 = new float[3*4];
		
		for (int i = 0; i < 12; i++) {
			matrix2[i] = i*i*2.678f;
		}
		
		columns1 = 2;
		rows1 = 2;
		columns2 = 2;
		rows2 = 6;
		
		matrix1Arg = IntArg.create(matrix1, true);
		matrix2Arg = HalfArg.create(matrix2, true);
	}
	
//	@Test
//	void testInvalidParameter(){
//		assertThrows(NullPointerException.class, () -> {
//			PaddingArg.createMatrixPadding(null, columns1, rows1, 4, 4, 0);
//		});
//		
//		assertThrows(IllegalArgumentException.class, () -> {
//			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, 4, -1, 0);
//		});
//		
//		assertThrows(IllegalArgumentException.class, () -> {
//			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, -1, 4, 0);
//		});
//		
//		assertThrows(IllegalArgumentException.class, () -> {
//			PaddingArg.createMatrixPadding(matrix1Arg, -1, rows1, 4, 4, 0);
//		});
//		
//		assertThrows(IllegalArgumentException.class, () -> {
//			PaddingArg.createMatrixPadding(matrix1Arg, columns1, Integer.MIN_VALUE, 4, 4, 0);
//		});
//		
//		assertThrows(IllegalArgumentException.class, () -> {
//			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, columns1, rows1, 0);
//		});
//		
//		assertThrows(IllegalArgumentException.class, () -> {
//			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, 4, rows1-1, 0);
//		});
//	}
	
	@Test
	void test() {
		System.out.println("\n");
		System.out.println("------------------------");
		IntArg out = IntArg.createOutput(4*4);
		System.out.println("Download?");
		System.out.println(matrix1Arg.isDownload());
		System.out.println("upload?");
		System.out.println(matrix1Arg.isUpload());
		PaddingArg in = PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, 4, 4, 1);
		System.out.println("Download?");
		System.out.println(in.isDownload());
		System.out.println("upload?");
		System.out.println(in.isUpload());
		Executor.launch(copyIntArrayString, "copyInt", 1, 4*4, in, out);
		
		assertArrayEquals(matrix1, matrix1Arg.asIntArray());
		System.out.println("OutArg:");
		System.out.println(Arrays.toString(out.asIntArray()));
		assertArrayEquals(new int[] {1,2,0,0, 3,4,0,0, 0,0,0,0, 0,0,0,0}, out.asIntArray());
		
		in = PaddingArg.createMatrixPadding(arg, columnsArg, rowsArg, columnsNew, rowsNew, paddingValue)
	}
}
