package nn.exp;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import nn.NeuralNet;
import nn.util.NeuralNetUtil;

public class Experiment2 {

    public static void main(String[] args) {

        NeuralNet nn = new NeuralNet(28 * 28, 100, 10, .3, NeuralNet::sigmoid);

        nn.init();

        for (int epoch = 0; epoch < 20; epoch++) {
            long time = NeuralNetUtil.train(nn);
            int errors = NeuralNetUtil.test(nn, false);
            System.out.printf("epoch:%02d train time: %02d minutes and %02d seconds. errors: %d\n", epoch + 1, (time / 1000) / 60, (time / 1000) % 60, errors);

            try (OutputStream os = new BufferedOutputStream(new FileOutputStream(String.format("epoch_%02d_%03d.nn", epoch + 1, errors)))) {
                nn.save(os);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
