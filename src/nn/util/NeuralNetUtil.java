package nn.util;

import static nn.tool.Converter.MAGIC_NUMBER_UBYTE;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

import javax.imageio.ImageIO;

import nn.NeuralNet;

public class NeuralNetUtil {

    public static String MNIST_DIR = "MNIST";

    public static String IMG_DIR = ".images";

    public static void save_image(String dir_name, String name, double[] data) {
        BufferedImage im = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int c = (int) ((data[x + y * 28] - .01) / .99 * 255.);
                im.setRGB(x, y, c | c << 8 | c << 16);
            }
        }

        try {
            File dir = new File(dir_name);
            if (!dir.exists())dir.mkdir();
            ImageIO.write(im, "PNG", new File(dir, name));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static long train(NeuralNet nn, int epoch) {
        long time = 0;

        for (int i = 0; i < epoch; i++) {
            time += train(nn);
        }

        return time; 
    }

    public static long train(NeuralNet nn) {
        byte[] arr = new byte[4];

        long time = System.currentTimeMillis();

        try (InputStream is = new BufferedInputStream(new FileInputStream(new File(MNIST_DIR, "train-ubyte")))) {

            is.read(arr);

            if (MAGIC_NUMBER_UBYTE != ByteBuffer.wrap(arr).getInt()) throw new RuntimeException();

            is.read(arr);
            int count = ByteBuffer.wrap(arr).getInt();

            is.read(arr);
            int rows = ByteBuffer.wrap(arr).getInt();

            is.read(arr);
            int colums = ByteBuffer.wrap(arr).getInt();

            double[] inputs = new double[rows * colums];

            for (int index = 0; index < count; index++) {
                int num = is.read();

                for (int j = 0; j < inputs.length; j++) {
                    inputs[j] = is.read() / 255. * .99 + .01;
                }

                double[] targets = new double[10];
                Arrays.fill(targets, .01);
                targets[num] = .99;

                nn.train(inputs, targets);
            }
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }

        return System.currentTimeMillis() - time;
    }

    public static int test(NeuralNet nn, boolean fix) {
        int err = 0;

        try (InputStream is = new BufferedInputStream(new FileInputStream(new File(MNIST_DIR, "t10k-ubyte")))) {
            byte[] arr = new byte[4];

            is.read(arr);

            if (MAGIC_NUMBER_UBYTE != ByteBuffer.wrap(arr).getInt()) throw new RuntimeException();

            is.read(arr);
            int count = ByteBuffer.wrap(arr).getInt();

            is.read(arr);
            int rows = ByteBuffer.wrap(arr).getInt();

            is.read(arr);
            int colums = ByteBuffer.wrap(arr).getInt();

            double[] inputs = new double[rows * colums];

            for (int index = 0; index < count; index++) {
                int num = is.read();

                for (int i = 0; i < inputs.length; i++) {
                    inputs[i] = is.read() / 255. * .99 + .01;
                }

                double[] outputs = nn.query(inputs);

                int max = 0;

                for (int i = 0; i < outputs.length; i++) {
                    max = outputs[i] > outputs[max] ? i : max;
                }

                if (num != max) {
                    err++;

                    if (fix) {
                        save_image(IMG_DIR, String.format("%05d_%d_%d.png", index, num, max), inputs);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }

        return err;
    }
}
