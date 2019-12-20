package yacx;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestProgram extends TestJNI {
	final static String fastMath = "--use_fast_math";
	final static String debug = "-G";
	final static String archOption = "--gpu-architecture";
	final static String archValue = "compute_35";
	final static String archValueInvalid = "compute_8000";
	
	static Options options, optionsInvalid;
	static String kernelInvalid, kernelInvalidName;

	@BeforeAll
	static void init() throws IOException {
		options = Options.createOptions();
		options.insert(fastMath);
		options.insert(debug);
		options.insert(archOption, archValue);
		
		optionsInvalid = Options.createOptions();
		optionsInvalid.insert(debug);
		optionsInvalid.insert(archOption, archValueInvalid);
		
		
		//Invalid kernel (very simalar to saxpy-kernel)
		kernelInvalid = "extern \"C\" __global__\n" + 
				"void saxpy(float a, float *x, float *y, float *out, size_t n) {\n" + 
				"  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n" + 
				"  if (tid < m) {\n" + //<- There is a m, not a n. It is not compilable!
				"    out[tid] = a * x[tid] + y[tid];\n" + 
				"  }\n" + 
				"}\n" + 
				"";
		
		kernelInvalidName = "saxpy";
	}
	
	@Test
	void testCreateInvalid() {
		//Check creating programs with null
		assertThrows(NullPointerException.class, () -> {
			Program.create(null, "saxpy");
		});
		
		assertThrows(NullPointerException.class, () -> {
			Program.create(saxpy, null);
		});
		
		assertThrows(NullPointerException.class, () -> {
			Program.create(null, null);
		});
	}
	
	@Test
	void testCreateValid() {
		//Create new Program
		Program p = Program.create(saxpy, "saxpy");
		
		assertNotNull(p);
		
		//Create other Program
		p = Program.create(filterk, "filter_k");
		
		assertNotNull(filterk);
	}
	
	@Test
	void testCompile() {
		Program saxpy = Program.create(TestProgram.saxpy, "saxpy");
		Program filter = Program.create(filterk, "filter_k");
		
		//Compile saxpy
		Kernel kernel = saxpy.compile();
		
		assertNotNull(kernel);
		
		//Compile saxpy again with same Program-Object
		kernel = saxpy.compile();
		
		assertNotNull(kernel);
		
		//Compile filter
		kernel = filter.compile();
		
		assertNotNull(kernel);
		
		//Compile invalid Program
		Program invaildProgram = Program.create(kernelInvalid, kernelInvalidName);
		assertThrows(ExecutorFailureException.class, () -> {
			invaildProgram.compile();
		});
	}
	
	@Test
	void testCompileOptions() {
		Program saxpy = Program.create(TestProgram.saxpy, "saxpy");
		Program filter = Program.create(filterk, "filter_k");
		
		//Compile saxpy with Null-Option
		assertThrows(NullPointerException.class, () -> {
			saxpy.compile(null);
		});
		
		//Compile saxpy and filter with correct options
		Kernel kernel = saxpy.compile(options);
		
		assertNotNull(kernel);
		
		kernel = filter.compile(options);
		
		assertNotNull(kernel);
		
		//Compile saxpy and filter with invalid options
		assertThrows(ExecutorFailureException.class, () -> {
			saxpy.compile(optionsInvalid);
		});
		
		assertThrows(ExecutorFailureException.class, () -> {
			filter.compile(optionsInvalid);
		});
		
		//Compile invalid Program
		Program invaildProgram = Program.create(kernelInvalid, kernelInvalidName);
		assertThrows(ExecutorFailureException.class, () -> {
			invaildProgram.compile(options);
		});
		
		assertThrows(ExecutorFailureException.class, () -> {
			invaildProgram.compile(optionsInvalid);
		});
	}

	@Test
	void testCreateHeaders() {
		//TODO
		//Issue #70
	}
}
