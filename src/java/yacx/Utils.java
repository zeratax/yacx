package yacx;

import java.io.IOException;
import java.io.File;
import java.nio.file.Files;

/**
 * Class with a useful function for reading a file.
 */
public class Utils {
	/**
	 * Reads a file and returns the content of the file as string.
	 * @param filename name of a file in current path or complete filename including path to file
	 * @return string with the filecontent
	 * @throws IOException
	 */
    public static String loadFile(String filename) throws IOException {
        assert(filename != null);
        
        return new String(Files.readAllBytes(new File(filename).toPath()));
    }
}
