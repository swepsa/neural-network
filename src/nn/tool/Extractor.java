package nn.tool;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import javax.imageio.ImageIO;

public class Extractor {

    public static int MAGIC_NUMBER_UBYTE = 0x4d4e5542; //MNUB

    public static String MNIST_DIR = "MNIST";
    public static String IMG_DIR = ".images";

    public static void save_image(String dir_name, int index, int num, byte[] data) {
        BufferedImage im = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int c = Byte.toUnsignedInt(data[x + y * 28]);
                int rgb = c | c << 8 | c << 16;
                im.setRGB(x, y, rgb);
            }
        }

        try {
            File dir = new File(dir_name);
            if (!dir.exists()) dir.mkdir();
            ImageIO.write(im, "PNG", new File(dir, String.format("%05d_%d.png", index, num)));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void extract(String name, int index, int count) {
        File ubyte = new File(MNIST_DIR, name);

        try (InputStream is = new BufferedInputStream(new FileInputStream(ubyte))) {

            byte[] arr = new byte[4];

            is.read(arr);
            if (MAGIC_NUMBER_UBYTE != ByteBuffer.wrap(arr).getInt()) throw new RuntimeException();

            is.read(arr);
            /*int count = ByteBuffer.wrap(arr).getInt(); */

            is.read(arr);
            int rows = ByteBuffer.wrap(arr).getInt();

            is.read(arr);
            int colums = ByteBuffer.wrap(arr).getInt();

            byte[] data = new byte[rows * colums];

            is.skip(index * (1 + data.length));

            for (int i = 0; i < count; i++) {
                int num = is.read();
                int read = is.read(data);
                if (read != data.length) throw new RuntimeException();
                save_image(IMG_DIR, index + i, num, data);
            }

        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }

    public static void main(String[] args) {
        extract("train-ubyte", 0, 20);
    }
}
