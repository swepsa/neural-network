package nn.exp;

import org.apache.commons.math3.util.FastMath;

import nn.NeuralNet;
import nn.util.NeuralNetUtil;

public class Experiment626 {

	public static String IMG_DIR = ".images"; 

	public static double logit(double x) {
        return FastMath.log(x / (1 - x));
    }

    public static void main(String[] args) {

        NeuralNet nn = new NeuralNet(28 * 28, 100, 10, .3, NeuralNet::sigmoid);

        nn.init();

        long time = NeuralNetUtil.train(nn, 1);

        System.out.printf("train time: %02d minutes and %02d seconds.\n", (time / 1000) / 60, (time / 1000) % 60);

        NeuralNetUtil.save_image(IMG_DIR, "0.png", nn.backquery(new double[] {.99, .01, .01, .01, .01, .01, .01, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "1.png", nn.backquery(new double[] {.01, .99, .01, .01, .01, .01, .01, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "2.png", nn.backquery(new double[] {.01, .01, .99, .01, .01, .01, .01, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "3.png", nn.backquery(new double[] {.01, .01, .01, .99, .01, .01, .01, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "4.png", nn.backquery(new double[] {.01, .01, .01, .01, .99, .01, .01, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "5.png", nn.backquery(new double[] {.01, .01, .01, .01, .01, .99, .01, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "6.png", nn.backquery(new double[] {.01, .01, .01, .01, .01, .01, .99, .01, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "7.png", nn.backquery(new double[] {.01, .01, .01, .01, .01, .01, .01, .99, .01, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "8.png", nn.backquery(new double[] {.01, .01, .01, .01, .01, .01, .01, .01, .99, .01}, Experiment626::logit));
        NeuralNetUtil.save_image(IMG_DIR, "9.png", nn.backquery(new double[] {.01, .01, .01, .01, .01, .01, .01, .01, .01, .99}, Experiment626::logit));
    }
}
