package nn.tool;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;

public class Converter {

    public static int MAGIC_NUMBER_LABEL = 0x00000801;
    public static int MAGIC_NUMBER_IMAGE = 0x00000803;

    public static int MAGIC_NUMBER_UBYTE = 0x4d4e5542; //MNUB

    public static String MNIST_DIR = "MNIST";

    private static void convert(String imeges_name, String labels_name, String ubytes_name) {
        File images_file = new File(MNIST_DIR, imeges_name);
        File labels_file = new File(MNIST_DIR, labels_name);
        File ubytes_file = new File(MNIST_DIR, ubytes_name);

        try (InputStream images = new BufferedInputStream(new FileInputStream(images_file));
             InputStream labels = new BufferedInputStream(new FileInputStream(labels_file));
             OutputStream ubytes = new BufferedOutputStream(new FileOutputStream(ubytes_file))) {

            byte[] arr = new byte[4];

            images.read(arr);
            int magic_num = ByteBuffer.wrap(arr).getInt();
            if (magic_num != MAGIC_NUMBER_IMAGE) throw new RuntimeException();

            ubytes.write(ByteBuffer.allocate(4).putInt(MAGIC_NUMBER_UBYTE).array());

            images.read(arr);
            int count = ByteBuffer.wrap(arr).getInt();

            ubytes.write(ByteBuffer.allocate(4).putInt(count).array());

            images.read(arr);
            int rows = ByteBuffer.wrap(arr).getInt();

            ubytes.write(ByteBuffer.allocate(4).putInt(rows).array());

            images.read(arr);
            int columns = ByteBuffer.wrap(arr).getInt();

            ubytes.write(ByteBuffer.allocate(4).putInt(columns).array());

            labels.read(arr);
            magic_num = ByteBuffer.wrap(arr).getInt();
            if (magic_num != MAGIC_NUMBER_LABEL) throw new RuntimeException();

            labels.read(arr);

            if (count != ByteBuffer.wrap(arr).getInt()) throw new RuntimeException();

            byte[] pixels = new byte[rows * columns];

            for (int index = 0; index < count; index++) {
                ubytes.write(labels.read());
                int n = images.read(pixels);
                if (n != pixels.length) throw new RuntimeException();
                ubytes.write(pixels);
            }

            ubytes.flush();
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }

    public static void main(String[] args) {
        convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "train-ubyte");
        convert("t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte",  "t10k-ubyte");
    }
}
