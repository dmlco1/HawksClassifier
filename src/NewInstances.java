import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NewInstances {
    Instances database = null;

    public NewInstances(Instances database) {
        database.clear();
        this.database = database;
    }

    public void addInstance(String[] attributes) {
        int nrAttr = database.numAttributes();
        if (nrAttr == attributes.length) {
            double[] instancesValue = new double[nrAttr];
            for (int i = 0; i < nrAttr; i++) {
                instancesValue[i] = database.attribute(i).indexOfValue(attributes[i]);
            }
            database.add(new DenseInstance(1.0, instancesValue));
        } else {
            System.out.println("Incorrect number of attributes");
        }
    }

    public Instances getDataset() {
        return database;
    }

    public static void main(String[] args) {
        DataSource source;
        try {
            source = new DataSource("pratica10.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1);

            NewInstances ni = new NewInstances(dataset);

            String[] values2 = {"TRUE","rainy", "unavailable", "go_to_beach"};

            ni.addInstance(values2);
            System.out.println(ni.getDataset());

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
}