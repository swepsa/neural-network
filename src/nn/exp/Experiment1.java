package nn.exp;

import nn.NeuralNet;
import nn.util.NeuralNetUtil;

public class Experiment1 {

    public static void main(String[] args) {
        NeuralNet nn = new NeuralNet(28 * 28, 100, 10, .3, NeuralNet::sigmoid);

        nn.init();

        long time = NeuralNetUtil.train(nn);

        System.out.printf("train time: %02d minutes and %02d seconds.\n\n", (time / 1000) / 60, (time / 1000) % 60);

        System.out.printf("errors: %d\n", NeuralNetUtil.test(nn, true));
    }
}
