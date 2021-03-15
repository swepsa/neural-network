package nn.exp;

import nn.NeuralNet;

public class Experiment0 {

    public static void main(String[] args) {
        NeuralNet nn = new NeuralNet(2, 4, 2, .2, NeuralNet::sigmoid);

        nn.init();

        long time = System.currentTimeMillis();

        for (int i = 0; i < 10000; i++) {
            nn.train(new double[] {.001, .001}, new double[] {.001, .001});
            nn.train(new double[] {.001, .999}, new double[] {.001, .999});
            nn.train(new double[] {.999, .001}, new double[] {.999, .001});
            nn.train(new double[] {.999, .999}, new double[] {.999, .999});

            nn.train(new double[] {.100, .100}, new double[] {.001, .001});
            nn.train(new double[] {.100, .800}, new double[] {.001, .999});
            nn.train(new double[] {.800, .100}, new double[] {.999, .001});
            nn.train(new double[] {.800, .800}, new double[] {.999, .999});

            nn.train(new double[] {.200, .200}, new double[] {.001, .001});
            nn.train(new double[] {.200, .700}, new double[] {.001, .999});
            nn.train(new double[] {.700, .200}, new double[] {.999, .001});
            nn.train(new double[] {.700, .700}, new double[] {.999, .999});

            nn.train(new double[] {.300, .300}, new double[] {.001, .001});
            nn.train(new double[] {.300, .600}, new double[] {.001, .999});
            nn.train(new double[] {.600, .300}, new double[] {.999, .001});
            nn.train(new double[] {.600, .600}, new double[] {.999, .999});

            nn.train(new double[] {.400, .400}, new double[] {.001, .001});
            nn.train(new double[] {.400, .500}, new double[] {.001, .999});
            nn.train(new double[] {.500, .400}, new double[] {.999, .001});
            nn.train(new double[] {.500, .500}, new double[] {.999, .999});

            nn.train(new double[] {.499, .499}, new double[] {.001, .001});
        }

        time = System.currentTimeMillis() - time;

        System.out.printf("train time: %d minutes and %d seconds.\n\n", (time / 1000) / 60, (time / 1000) % 60);

        double[] r;

        r = nn.query(new double[] {.001, .020}); System.out.printf("%f %f\n", r[0], r[1]); // 0 0
        r = nn.query(new double[] {.090, .510}); System.out.printf("%f %f\n", r[0], r[1]); // 0 1
        r = nn.query(new double[] {.690, .410}); System.out.printf("%f %f\n", r[0], r[1]); // 1 0
        r = nn.query(new double[] {.770, .770}); System.out.printf("%f %f\n", r[0], r[1]); // 1 1
    }
} // class
