import java.io.IOException;
import java.io.File;
import java.nio.file.Files;

public class Utils {
    public static String loadFile(String filename) throws IOException {
        assert(filename != null);
        return new String(Files.readAllBytes(new File(filename).toPath()));
    }
}
