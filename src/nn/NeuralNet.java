package nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

//Класс нейронной сети
public class NeuralNet {

    public interface ActivationFunction {
        double value(double x);
    }

    // Сигмоида
    public static double sigmoid(double x) {
        return 1 / (1 + FastMath.exp(-x));
    }

    private double learning_rate;

    private ActivationFunction activation_function;

    private RealMatrix wih, who;

    private NeuralNet(RealMatrix wih, RealMatrix who, double learning_rate, ActivationFunction activation_function) {
        this.wih = wih;
        this.who = who;
        this.learning_rate = learning_rate;

        this.activation_function = activation_function;
    }

    public NeuralNet(int inputnodes, int hiddennodes, int outputnodes, double learning_rate, ActivationFunction activation_function) {
        this.learning_rate = learning_rate;
        this.activation_function = activation_function;

        wih = new Array2DRowRealMatrix(hiddennodes, inputnodes);
        who = new Array2DRowRealMatrix(outputnodes, hiddennodes);
    }

    private void rand(RealMatrix matrix, Random rnd, double scale) {
        for (int row = 0; row < matrix.getRowDimension(); row++) {
            for (int column = 0; column < matrix.getColumnDimension(); column++) {
                matrix.setEntry(row, column, ThreadLocalRandom.current().nextDouble(-scale, scale));
            }
        }
    }

    public void init() {
        final Random rnd = new Random(System.currentTimeMillis());
        rand(wih, rnd, FastMath.pow(wih.getRowDimension(), -.5));
        rand(who, rnd, FastMath.pow(who.getRowDimension(), -.5));
    }

    // расчёт новых весов
    private RealMatrix weight_correction(RealMatrix weight, RealMatrix input, RealMatrix outputs, RealMatrix errors) {
        for (int row = 0; row < errors.getRowDimension(); row++) {
            for (int column = 0; column < errors.getColumnDimension(); column++) {
                errors.setEntry(row, column, errors.getEntry(row, column) * outputs.getEntry(row, column) * (1.0 - outputs.getEntry(row, column)));
            }
        }

        return weight.add(errors.multiply(input.transpose()).scalarMultiply(learning_rate));
    }

    public void train(double[] inputs, double[] targets) {
        // преобразовать массивы входных значений в матрицы
        RealMatrix inputs_matrix  = new Array2DRowRealMatrix(inputs);
        RealMatrix targets_matrix = new Array2DRowRealMatrix(targets);
        // рассчитать входящие сигналы для скрытого слоя
        RealMatrix hidden_outputs = wih.multiply(inputs_matrix);
        // рассчитать исходящие сигналы для скрытого слоя
        for (int row = 0; row < hidden_outputs.getRowDimension(); row++) { 
            hidden_outputs.setEntry(row, 0, activation_function.value(hidden_outputs.getEntry(row, 0)));
        }
        // рассчитать входящие сигналы для выходного слоя
        RealMatrix final_outputs = who.multiply(hidden_outputs);
        // рассчитать исходящие сигналы для выходного слоя
        for (int row = 0; row < final_outputs.getRowDimension(); row++) {
            final_outputs.setEntry(row, 0, activation_function.value(final_outputs.getEntry(row, 0)));
        }
        // ошибки выходного слоя = (целевое значение - фактическое значение)
        RealMatrix output_errors = targets_matrix.subtract(final_outputs);
        // ошибки скрытого слоя - это ошибки output_errors,
        // распределенные пропорционально весовым коэффициентам связей
        // и рекомбинированные на скрытых узлах
        RealMatrix hidden_errors = who.transpose().multiply(output_errors);
        // обновить веса для связей между скрытым и выходным слоями
        who = weight_correction(who, hidden_outputs, final_outputs, output_errors);
        // обновить весовые коэффициенты для связей между входным и скрытым слоями
        wih = weight_correction(wih, inputs_matrix, hidden_outputs, hidden_errors);
    }

    // опрос нейронной сети
    public double[] query(double[] inputs) {
        // преобразовать массив входных значений в матрицу
        RealMatrix matrix = new Array2DRowRealMatrix(inputs);
        // рассчитать входящие сигналы для скрытого слоя
        matrix = wih.multiply(matrix);
        // рассчитать исходящие сигналы для скрытого слоя
        for (int row = 0; row < matrix.getRowDimension(); row++) { 
            matrix.setEntry(row, 0, activation_function.value(matrix.getEntry(row, 0)));
        }
        // рассчитать входящие сигналы для выходного слоя
        matrix = who.multiply(matrix);
        // рассчитать исходящие сигналы для выходного слоя
        for (int row = 0; row < matrix.getRowDimension(); row++) {
            matrix.setEntry(row, 0, activation_function.value(matrix.getEntry(row, 0)));
        }
        // вернуть результат запроса
        return matrix.getColumn(0);
    }

    // поиск минимального значения
    private double min(RealMatrix matrix) {
        int minAt = 0;

        for (int row = 0; row < matrix.getRowDimension(); row++) {
            minAt = matrix.getEntry(row, 0) < matrix.getEntry(minAt, 0) ? row : minAt;
        }

        return matrix.getEntry(minAt, 0);
    }

    // поиск махсимального значения
    private double max(RealMatrix matrix) {
        int maxAt = 0;

        for (int row = 0; row < matrix.getRowDimension(); row++) {
            maxAt = matrix.getEntry(row, 0) > matrix.getEntry(maxAt, 0) ? row : maxAt;
        }

        return matrix.getEntry(maxAt, 0);
    }

    public double[] backquery(double[] targets, ActivationFunction inverse_activation_function) {
        // преобразовать массив выходных значений в матрицу
        RealMatrix final_outputs = new Array2DRowRealMatrix(targets);

        RealMatrix final_inputs = final_outputs;
        // calculate the signal into the final output layer
        for (int row = 0; row < final_inputs.getRowDimension(); row++) {
            final_inputs.setEntry(row, 0, inverse_activation_function.value(final_inputs.getEntry(row, 0)));
        }
        // calculate the signal out of the hidden layer
        RealMatrix hidden_outputs = who.transpose().multiply(final_inputs);
        // scale them back to 0.01 to .99
        double min = min(hidden_outputs);

        for (int row = 0; row < hidden_outputs.getRowDimension(); row++) {
            hidden_outputs.setEntry(row, 0, hidden_outputs.getEntry(row, 0) - min);
        }

        double max = max(hidden_outputs);

        for (int row = 0; row < hidden_outputs.getRowDimension(); row++) {
            hidden_outputs.setEntry(row, 0, hidden_outputs.getEntry(row, 0) / max);
        }

        hidden_outputs = hidden_outputs.scalarMultiply(0.98).scalarAdd(0.01);

        // calculate the signal into the hidden layer
        RealMatrix hidden_inputs = hidden_outputs;

        for (int row = 0; row < hidden_inputs.getRowDimension(); row++) {
            hidden_inputs.setEntry(row, 0, inverse_activation_function.value(hidden_inputs.getEntry(row, 0)));
        }

        // calculate the signal out of the input layer
        RealMatrix inputs = wih.transpose().multiply(hidden_inputs);

        // scale them back to 0.01 to .99
        min = min(inputs);

        for (int row = 0; row < inputs.getRowDimension(); row++) {
            inputs.setEntry(row, 0, inputs.getEntry(row, 0) - min);
        }

        max = max(inputs);

        for (int row = 0; row < inputs.getRowDimension(); row++) {
            inputs.setEntry(row, 0, inputs.getEntry(row, 0) / max);
        }

        inputs = inputs.scalarMultiply(0.98).scalarAdd(0.01);

        return inputs.getColumn(0);
    }

    public void save(OutputStream os) throws IOException {
        DataOutputStream dos = new DataOutputStream(os);

        dos.writeDouble(learning_rate);

        dos.writeInt(wih.getRowDimension());
        dos.writeInt(wih.getColumnDimension());

        for (int row = 0; row < wih.getRowDimension(); row++) {
            for (int column = 0; column < wih.getColumnDimension(); column++) {
                dos.writeDouble(wih.getEntry(row, column));
            }
        }

        dos.writeInt(who.getRowDimension());
        dos.writeInt(who.getColumnDimension());

        for (int row = 0; row < who.getRowDimension(); row++) {
            for (int column = 0; column < who.getColumnDimension(); column++) {
                dos.writeDouble(who.getEntry(row, column));
            }
        }

        dos.flush();
    }

    public static NeuralNet load(InputStream is, ActivationFunction activation_function) throws IOException {
        DataInputStream dis = new DataInputStream(is);

        double learning_rate = dis.readDouble();

        RealMatrix wih = new Array2DRowRealMatrix(dis.readInt(), dis.readInt());

        for (int row = 0; row < wih.getRowDimension(); row++) {
            for (int column = 0; column < wih.getColumnDimension(); column++) {
                wih.setEntry(row, column, dis.readDouble());
            }
        }

        RealMatrix who = new Array2DRowRealMatrix(dis.readInt(), dis.readInt());

        NeuralNet nn = new NeuralNet(wih, who, learning_rate, activation_function);

        for (int row = 0; row < who.getRowDimension(); row++) {
            for (int column = 0; column < who.getColumnDimension(); column++) {
                who.setEntry(row, column, dis.readDouble());
            }
        }

        return nn;
    }
} // class
