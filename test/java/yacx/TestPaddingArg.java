package yacx;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

@TestMethodOrder(OrderAnnotation.class)
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
	
	static final String addIntArrayString = "extern \"C\" __global__\n" + 
			"void addInt(int* sum1, int* sum2) {\n" + 
			"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
			"  sum1[i] += sum2[i];\n" + 
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
		//TODO use matrix2Arg
//		matrix2Arg = HalfArg.create(matrix2, true);
	}
	
	@Test
	@Order(1)
	void testInvalidParameter(){
		assertThrows(NullPointerException.class, () -> {
			PaddingArg.createMatrixPadding(null, columns1, rows1, 4, 4, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, 4, -1, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, -1, 4, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			PaddingArg.createMatrixPadding(matrix1Arg, -1, rows1, 4, 4, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			PaddingArg.createMatrixPadding(matrix1Arg, columns1, Integer.MIN_VALUE, 4, 4, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, columns1, rows1, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, 4, rows1-1, 0);
		});
	}
	
	@Test
	@Order(2)
	void testCopy() {
		assertTrue(matrix1Arg.isDownload());
		
		IntArg out = IntArg.createOutput(4*4);
		PaddingArg in = PaddingArg.createMatrixPadding(matrix1Arg, columns1, rows1, 4, 4, 1);
		
		assertTrue(matrix1Arg.isDownload());
		assertTrue(in.isDownload());
		assertEquals(4*4, in.getLength());
		assertEquals(matrix1Arg, in.getArg());
		
		//copy matrix1 with 1-padding to 4x4 matrix
		Executor.launch(copyIntArrayString, "copyInt", 1, 4*4, in, out);
		
		//original matrix should be unchanged
		assertEquals(matrix1.length, matrix1Arg.getLength());
		assertArrayEquals(matrix1, matrix1Arg.asIntArray());
		//matrix with padding
		assertArrayEquals(new int[] {1,2,1,1, 3,4,1,1, 1,1,1,1, 1,1,1,1}, out.asIntArray());
		
		
		out = IntArg.createOutput(4*5);
		in = PaddingArg.createMatrixPadding(matrix1Arg, rows1, columns1, 4, 5, 0);
		
		assertTrue(matrix1Arg.isDownload());
		assertTrue(in.isDownload());
		assertEquals(4*5, in.getLength());
		assertEquals(matrix1Arg, in.getArg());
		
		//copy matrix1 with 0-padding to 4x5 matrix
		Executor.launch(copyIntArrayString, "copyInt", 1, 4*5, in, out);
		
		//original matrix should be unchanged
		assertArrayEquals(matrix1, matrix1Arg.asIntArray());
		//matrix with padding
		assertArrayEquals(new int[] {1,2,0,0,0, 3,4,0,0,0, 0,0,0,0,0, 0,0,0,0,0,}, out.asIntArray());
		
		
		matrix1Arg.setDownload(false);
		out = IntArg.createOutput(5*4);
		in = PaddingArg.createMatrixPadding(matrix1Arg, rows1, columns1, 5, 4, -1);
		
		assertFalse(matrix1Arg.isDownload());
		assertFalse(in.isDownload());
		assertEquals(5*4, in.getLength());
		assertEquals(matrix1Arg, in.getArg());
		
		//copy matrix1 with -1-padding to 5*4 matrix
		Executor.launch(copyIntArrayString, "copyInt", 1, 5*4, in, out);
		
		//original matrix should be unchanged
		assertArrayEquals(matrix1, matrix1Arg.asIntArray());
		//matrix with padding
		assertArrayEquals(new int[] {1,2,-1,-1, 3,4,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1}, out.asIntArray());
	}
	
	@Test
	@Order(3) //this test change matrix1Arg
	void testAdd() {
		matrix1Arg.setDownload(true);
		assertTrue(matrix1Arg.isDownload());

		PaddingArg in = PaddingArg.createMatrixPadding(matrix1Arg, rows1, columns1, 3, 5, 0);
		IntArg sum2 = IntArg.create(1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1);
		
		assertTrue(matrix1Arg.isDownload());
		assertTrue(in.isDownload());
		assertEquals(3*5, in.getLength());
		
		//copy matrix1 with 0-padding to 4x5 matrix
		Executor.launch(addIntArrayString, "addInt", 1, 3*5, in, sum2);
		
		//original matrix should be changed
		assertEquals(matrix1.length, matrix1Arg.getLength());
		for (int i = 0; i < matrix1.length; i++) {
			assertEquals(matrix1[i]+1, matrix1Arg.asIntArray()[i]);
		}
		
		
		PaddingArg sum22 = PaddingArg.createMatrixPadding(matrix1Arg, rows1, columns1, 2, 6, 3);
		IntArg sum12 = IntArg.create(new int[] {1,1,1,1,1,1, 2,2,2,2,2,2}, true);
		
		assertTrue(matrix1Arg.isDownload());
		assertTrue(sum22.isDownload());
		assertEquals(2*6, sum22.getLength());
		
		//copy matrix1 with 0-padding to 4x5 matrix
		Executor.launch(addIntArrayString, "addInt", 1, 2*6, sum12, sum22);
		
		//original matrix should be unchanged
		for (int i = 0; i < matrix1.length; i++) {
			assertEquals(matrix1[i]+1, matrix1Arg.asIntArray()[i]);
		}
		
		//check sum2 matrix
		assertEquals(matrix1[0]+1+1, sum12.asIntArray()[0]);
		assertEquals(matrix1[1]+1+1, sum12.asIntArray()[1]);
		assertEquals(3+1, sum12.asIntArray()[2]);
		assertEquals(3+1, sum12.asIntArray()[3]);
		assertEquals(3+1, sum12.asIntArray()[4]);
		assertEquals(3+1, sum12.asIntArray()[5]);
		
		assertEquals(matrix1[2]+1+2, sum12.asIntArray()[6]);
		assertEquals(matrix1[3]+1+2, sum12.asIntArray()[7]);
		assertEquals(3+2, sum12.asIntArray()[8]);
		assertEquals(3+2, sum12.asIntArray()[9]);
		assertEquals(3+2, sum12.asIntArray()[10]);
		assertEquals(3+2, sum12.asIntArray()[11]);
	}
}
