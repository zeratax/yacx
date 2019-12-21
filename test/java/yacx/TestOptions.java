package yacx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TestOptions extends TestJNI {
	final String fastMath = "--use_fast_math";
	final String debug = "-G";
	final String archOption = "--gpu-architecture";
	final String archValue = "compute_35";
	final String arch1= archOption + "=" + archValue;
	final String language = "--std";
	final String languageValue = "c++14";
	final String language1 = language + "=" + languageValue;
	
	Options options;
	
	@Test
	void testConstructorInvalid() {
		//Test Null
		assertThrows(NullPointerException.class, () -> {
			Options.createOptions((String) null);
		});
		
		assertThrows(NullPointerException.class, () -> {
			Options.createOptions((String[]) null);
		});
		
		assertThrows(NullPointerException.class, () -> {
			Options.createOptions(debug, null, arch1);
		});
		
		assertThrows(NullPointerException.class, () -> {
			Options.createOptions(new String[] {debug, null, archOption});
		});
	}
	
	@Test
	void testConstructorValid() {
		//Test constructor without parameters
		options = Options.createOptions();
				
		assertNotNull(options);
		assertEquals(0, options.getSize());
		assertArrayEquals(new String[0], options.getOptions());		
		
		//Test constructor with 1 parameter
		options = Options.createOptions(debug);
		
		assertNotNull(options);
		assertEquals(1, options.getSize());
		assertArrayEquals(new String[] {debug}, options.getOptions());
		
		//Test constructor with 2 parameters
		options = Options.createOptions(debug, arch1);
		
		assertNotNull(options);
		assertEquals(2, options.getSize());
		assertArrayEquals(new String[] {debug, arch1}, options.getOptions());
		
		//Test constructor with 3 parameters as String[]
		String[] optionsString = new String[] {debug, fastMath, arch1};
		options = Options.createOptions(optionsString);
		
		assertNotNull(options);
		assertEquals(3, options.getSize());
		assertArrayEquals(new String[] {debug, fastMath, arch1}, options.getOptions());
	}

	@Test
	void testInsert() {
		//Create empty options
		options = Options.createOptions();
		
		//Test Null
		assertThrows(NullPointerException.class, () -> {
			options.insert(null);
		});
		
		assertNotNull(options);
		assertEquals(0, options.getSize());
		assertArrayEquals(new String[0], options.getOptions());	
		
		//insert 1 option
		options.insert(debug);
		
		assertEquals(1, options.getSize());
		assertArrayEquals(new String[] {debug}, options.getOptions());	
		
		//Insert second option
		options.insert(arch1);
		
		assertEquals(2, options.getSize());
		assertArrayEquals(new String[] {debug, arch1}, options.getOptions());	
	}

	@Test
	void testInsertValue() {
		//Create options with fast-math
		options = Options.createOptions(fastMath);
		
		//Test Null
		assertThrows(NullPointerException.class, () -> {
			options.insert(null, fastMath);
		});
				
		assertThrows(NullPointerException.class, () -> {
			options.insert(fastMath, null);
		});
		
		assertThrows(NullPointerException.class, () -> {
			options.insert(null, null);
		});
		
		assertNotNull(options);
		assertEquals(1, options.getSize());
		assertArrayEquals(new String[] {fastMath}, options.getOptions());	
		
		//insert second option
		options.insert(archOption, archValue);
		
		assertNotNull(options);
		assertEquals(2, options.getSize());
		assertArrayEquals(new String[] {fastMath, arch1}, options.getOptions());	
		
		//Insert third option
		options.insert(language, languageValue);
		
		assertNotNull(options);
		assertEquals(3, options.getSize());
		assertArrayEquals(new String[] {fastMath, arch1, language1}, options.getOptions());	
	}
}
