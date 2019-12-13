import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.reflect.Field;

import org.junit.jupiter.api.Test;

class TestJNIHandle extends TestJNI {
	
	/**
	 * Check handle-member not 0 and after destroying c-object the handle-member should be 0 
	 * @param jni JNI-Object
	 */
	void checkAndDispose(JNIHandle jni) throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		assertTrue(getHandle(jni) != 0);
		jni.dispose();
		assertFalse(getHandle(jni) != 0);
	}
	
	/**
	 * Get handle-member from JNI-Object
	 * @param jni JNI-Object
	 * @return returns private member handle from JNI-Object
	 */
	long getHandle(JNIHandle jni) throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		Field handleField = JNIHandle.class.getDeclaredField("nativeHandle");

		handleField.setAccessible(true);
		
		long handleValue = (Long) handleField.get(jni);
		
		return handleValue;
	}

//	@Test
	void testDispose() throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException, IOException {
		//Check KernalArgs
		checkAndDispose(BooleanArg.create(false));
		checkAndDispose(ByteArg.create((byte) 7));
		checkAndDispose(ShortArg.create((short) 6));
		checkAndDispose(IntArg.create(6));
		checkAndDispose(LongArg.create(7l, 6l));
		checkAndDispose(FloatArg.create(new float[] {1, 2f}));
		checkAndDispose(DoubleArg.create(4.7d, 6.8d));
		
		//Check Device, Options and Headers
		checkAndDispose(Device.createDevice());
		checkAndDispose(Options.createOptions());
		checkAndDispose(Headers.createHeaders());
		
		//Check Program and Kernel
		
		Program p0 = Program.create(Utils.loadFile("saxpy.cu"), "saxpy");
		//Create a new Program-Object pointing to the same C-Object :)
		long handleP = getHandle(p0);
		Program p = new Program(handleP);
		//Create a thrid Program-Object with same strings
		Program p2 = Program.create(Utils.loadFile("saxpy.cu"), "saxpy");
		
		//p0 and p should be equal
		assertEquals(p0, p);
		//p0 and p2 should be different cause they pointing to different C-Objects
		assertNotEquals(p0, p2);
		
		Kernel saxpyKernel = p.compile();
		
		checkAndDispose(saxpyKernel);
		checkAndDispose(p);
		
		//C-Object from Program p should be destroyed
		assertThrows(IllegalArgumentException.class, () -> {
			p.compile(); //TODO Segfault
		});
		
		//P2 should be valid stayed
		saxpyKernel = p2.compile();
	}
}
