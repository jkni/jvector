package io.github.jbellis.jvector.vector.cnative;

import java.io.File;
import java.nio.file.Files;

/**
 * This class is used to load the native library. First, it tries to load the library from the system path.
 * If that fails, it tries to load the library from the classpath (using the usual copying to a tmp directory route).
 */
public class LibraryLoader {
    private LibraryLoader() {}
    public static boolean loadJvector() {
        try {
            System.loadLibrary("jvector");
            return true;
        } catch (UnsatisfiedLinkError e) {
            // ignore
        }
        try {
            // we don't want to import any libraries, so we'll just use the classloader to load the library as a resource
            // and then copy it to a tmp directory and load it from there
            String libName = System.mapLibraryName("jvector");
            // create a tempFile with libName split into prefix/suffix
            File tmpLibFile = File.createTempFile(libName.substring(0, libName.lastIndexOf('.')), libName.substring(libName.lastIndexOf('.')));
            // open libName resource as inputstream, tmpLibFile as bufferedoutputstream
            try (var in = LibraryLoader.class.getResourceAsStream("/" + libName);
                 var out = Files.newOutputStream(tmpLibFile.toPath())) {
                    in.transferTo(out);
                    out.flush();
            }
            System.load(tmpLibFile.getAbsolutePath());
            return true;
        } catch (Exception | UnsatisfiedLinkError e) {
            // ignore
        }
        return false;
    }

}
