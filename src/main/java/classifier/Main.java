 package classifier;



import java.io.*;
import static java.lang.System.exit;
import java.util.*;
import utility.IOManager;
import static utility.IOManager.writeToCsvFile;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;


public class Main {
    
    private static final String tweetsCSV = "file/tweets.csv";// "file/testsnellito.csv";
    private static final String stopWordFile = "file/stopWord";
    private static final String stemmerCSV = "file/stemmer.csv";
    
    private static int numberTweets;
    private static String[] tweets;
    private static String[] classes;
    
    private static HashMap<String, Double> attributeWeights; // Peso per ogni features selezionata
    private static List<Attribute> attributes; // Features selezionate
    private static AttributeSelection attSelect;
    private static Double classificatorAccuracy;
    
    

    
    public static void main(String[] args) throws IOException, Exception {
        System.out.println("Main 1");
        
        //Import Tweets
        List<String[]> classList = IOManager.readFromCsvFile(";",tweetsCSV);
        System.out.println(classList.get(0).length);
        
        numberTweets = classList.size();
        tweets = new String[numberTweets];
        classes = new String[numberTweets];
        FeaturesExtractor fe = new FeaturesExtractor();
        
        int tweetNumber = 0;
        //Concatenate the 5 bit of class
        for(String[] s : classList){
            String tmp = "";
            for(int i = 0; i<5; i++){
                tmp = tmp + s[i];
            }
            classes[tweetNumber] = tmp;
            tweets[tweetNumber] = s[5]; 
            tweetNumber++;
        } 

        //Preprocessing
        tweets = fe.deleteUrlAndPic(tweets);    
        
        List<List<String>> tokenized = fe.tokenization(tweets);
        
        List<String> stopWords =  IOManager.readFromFile(stopWordFile);
        List<List<String>> tweetsNoStopWord = fe.stopWordFiltering(stopWords, tokenized);
        
        List<String[]> stemmingList = IOManager.readFromCsvFile(";",stemmerCSV);
        List<List<String>> tweetsStemmed = fe.stemming(stemmingList, tweetsNoStopWord);
        
        //Extract all attributes
        String[] allAttributes = fe.extractAllAttributes(tweetsStemmed);
        
        //Build structure to convert in csv
        List<String[]> allTweets = fe.getArrayAllTweets(allAttributes, classes, tweetsStemmed);
        writeToCsvFile(allTweets,",","allTweets.csv");
        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        DataSource source = new DataSource("allTweets.csv");
        
        Instances data = source.getDataSet();
        
        // Select class attribute
        data.setClassIndex(data.numAttributes() - 1);  

        //Balancing dataset
        data = balancingDataset(data);
        
        //SMOTE smote = new SMOTE();
	//smote.setInputFormat(data);
        
        System.out.println("# classi: " + data.numClasses());
        System.out.println("# attributi: " + data.numAttributes());
        System.out.println("# istanze: " + data.numInstances());
        
        Classifier cl =  null;
        Double nu = 0.79;
        int k = 1;
        Double MaxAcc= 0.0;
        String max = "";
            for(; nu<0.8; nu+=0.1){
                for(int n = 2000; n<2050; n+=100){
                    //for(; k<6; k++){
                        cl = crossFoldValidation(data,10,"svm",nu,k,n); // Folds number - "tree" or "svm" or "knn" -- nu for nu-svm and k for knn -- "n" # di features da selezionare
                        if(classificatorAccuracy>MaxAcc){
                            MaxAcc = classificatorAccuracy;
                            max = "svm => n: "+n+" nu: "+nu+" ==> ";
                        }
                //}
                }}
            System.out.println(max + MaxAcc);
        //knn non va !!!!!!!!!!!!!!!!!!!
        //cl = standardClassification(data,"svm"); // "tree" or "svm" or "knn"   
        
       // weka.core.SerializationHelper.write("/classifier/cl.model", cl);
       //Classifier cls = (Classifier) weka.core.SerializationHelper.read("/some/where/j48.model")
       
       String name = "treeCV";
       Classify classify = new Classify(name,classificatorAccuracy,cl,attributeWeights,attributes,attSelect);
       String file = "classifier/"+name+".classifier";
       classify.save(file);
       
       Classify a = new Classify(file);
       System.out.println(a.getName()+" "+a.getAccuracy());
    }
    
    

    // SUL PRIMO DA UN ERRORE, NON LO LEGGE

    private static Instances[] getTrainAndTestSet(Instances data) {
        data.randomize(new java.util.Random());	// randomize instance order before splitting dataset
        Instances train = data.trainCV(5, 0);
        Instances test = data.testCV(5, 0);
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);  

        
        // Elimino tutti gli attributi che non sono presenti nel training set
        // Remove last attribute from the counting, is a class
        for(int i=0; i<train.numAttributes()-1;) {
            //Se la media è 0, nessun tweet contiene quell'attributo (scansione verticale tra tutti i tweets)
            if(train.meanOrMode(train.attribute(i)) == 0){  
                train.deleteAttributeAt(i);
                test.deleteAttributeAt(i);
                // Non incremento i se lo elimino perchè il prossimo attributo prenderà quell'indice
            }else{
                i++;
            }           
        }
        Instances[] ret = {train,test};  
        return ret;
    }
    

    private static Instances featuresSelection(Instances data, int n) throws Exception{
         //Estraggo la lista degli attributi e conto ognuno in quante SUM è presente        
       
         attributes =  new ArrayList<>();
         attributeWeights = new  HashMap<>();
  
        // Remove last attribute from the counting, is a class
        for(int i=0; i<data.numAttributes()-1; i++) {
            attributes.add(data.attribute(i));
        }
        

        Double value;
        for(Instance inst : data){
            for(Attribute attr : attributes){ 
                if(inst.value(attr) == 1){
                    if(!attributeWeights.containsKey(attr.name())){
                        attributeWeights.put(attr.name(),new Double(1));
                    } else {
                        value = attributeWeights.get(attr.name());
                        attributeWeights.replace(attr.name(),++value);
                    }
                }      
            }
        }
        
        // Calcolo i pesi per ogni attributo del training set
        int numberOfInstances = data.numInstances();
        for(Map.Entry<String, Double> entry: attributeWeights.entrySet()) {
            attributeWeights.replace(entry.getKey(),Math.log(numberOfInstances/entry.getValue()));
        }
        
        // Update value of stem in train with pesi
        int pos = 0;
        for(Instance inst : data){
            for(Attribute attr : attributes){ 
                if(inst.value(attr) == 1){
                    inst.setValue(attr, attributeWeights.get(attr.name()));                    
                }
            } 
            data.set(pos++, inst);
        }    
        
        //System.out.println(train);
        //int n = 1650; //TODO Come selgo il numero di tweets da tenere?
        // CALCOLO INFOGAIN
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
	Ranker search = new Ranker();
	//search.setOptions(new String[] { "-T", "0.0001" });	// information gain threshold
	search.setNumToSelect(n-1); // Seleziono i primi n elementi
        attSelect = new AttributeSelection();
	attSelect.setEvaluator(eval);
	attSelect.setSearch(search);


	// apply attribute selection
	attSelect.SelectAttributes(data);
   
        
        
	// remove the attributes not selected in the last run
	data = attSelect.reduceDimensionality(data);
        return data;
    }
    
    
    private static Instances selectCorrectFeatures(Instances data) throws Exception{
        if(attributes == null || attributeWeights == null || attSelect == null){
            System.out.println("Ateenzione, prima di 'selectCorrectFeatures', usare 'featuresSelection'");
            exit(1);
        }
            
        int pos = 0;
        for(Instance inst : data){
            for(Attribute attr : attributes){ 
                if(inst.value(attr) == 1){
                    inst.setValue(attr, attributeWeights.get(attr.name()));                    
                }
            } 
            data.set(pos++, inst);
        }
        data = attSelect.reduceDimensionality(data);
        
        return data;
    }

    private static Instances balancingDataset(Instances data) throws Exception {
               //Balancing dataset
        SpreadSubsample ff = new SpreadSubsample();

        String opt = "-M 1.0 -X 0.0 -S 1";//any options you like, see documentation
        String[] optArray = weka.core.Utils.splitOptions(opt);//right format for the options

        ff.setOptions(optArray);
        ff.setInputFormat(data);


        return Filter.useFilter(data, ff);
    }

    private static J48 treeClassifier(Instances train) throws Exception {
        // Building the classifier
            String [] options =new String[1];
            options[0] = "-U";
            J48 tree = new J48();
            tree.setOptions(options);
            tree.buildClassifier(train);
            // Evaluation on the training set            
            return tree;
    }

    private static LibSVM SVMClassifier(Instances train, int c, int d, Double g, int k, int s, Double nu) throws Exception {
        LibSVM svm = new LibSVM();
        // Sono opzioni prese da internet, verificare
        String options = ( "-S "+s+" -K "+k+" -D "+d+" -G "+g+" -R 0.0 -N "+nu+" -M 40.0 -C "+c+" -E 0.001 -P 0.1" );
        String[] optionsArray = options.split( " " );
        svm.setOptions( optionsArray );
        svm.buildClassifier(train);
        return svm;
    }

    private static Classifier crossFoldValidation(Instances data, int folds, String classificatorType, Double nu, int k, int n) throws Exception {
        data = featuresSelection(data,n);
        
        //Training instances are held in "originalTrain"
        Classifier cl = null;
        
        if(classificatorType.equals("tree")){
            cl = new J48();
        } else if(classificatorType.equals("svm")){
            LibSVM svm = new LibSVM();
            String options = ( "-S 1 -K 0 -D 1 -G 0.1 -R 0.0 -N "+nu+" -M 40.0 -C 1 -E 0.001 -P 0.1" );
            String[] optionsArray = options.split( " " );
            svm.setOptions( optionsArray );
            cl = svm;
        } else if(classificatorType.equals("knn")){
            IBk ibk = new IBk();
            String options = ( "-K "+k);
            String[] optionsArray = options.split( " " );
            ibk.setOptions( optionsArray );
            cl = ibk;
        } else{
            System.out.println("Nome classificatore errato");
            exit(1);
        }
        
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cl, data, folds, new Random(1));
        System.out.println("NU: "+nu+" K: "+k);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.pctCorrect());
        System.out.println("Estimated Accuracy: "+Double.toString(eval.pctCorrect()));
        for(int i= 0;i< data.numClasses(); i++){
            System.out.println("Sensitivity(TP/P) class "+i+": "+eval.truePositiveRate(i));
            System.out.println("Specificity(TN/N) class "+i+": "+eval.trueNegativeRate(i));
        }
        classificatorAccuracy = eval.pctCorrect();
        return cl;
    }

    private static Classifier standardClassification(Instances data, String classificatorType) throws Exception {
        //Train a new classifier
        Classifier c2 = new J48();
        c2.buildClassifier(data);  //predict with this model
        Classifier classificator = null;


        int s=1;
        int k=0;
        HashMap<String, Double> accuracy = new  HashMap<>(); 
        for(int iteration = 0; iteration<30; iteration++){
            // Split into Test and Training set, messo qua perchè ogni volta gli elementi che compongono i set sono scelti randomicamente
            Instances[]sets = getTrainAndTestSet(data);
            Instances train = sets[0];
            Instances test = sets[1];
            for(int n = 1900; n<2000; n+=100){

                // Calcolo i pesi e seleziono gli attributi mediante infogain
                train = featuresSelection(train,n);
                test = selectCorrectFeatures(test); 

                for(int c = 1; c<5000; c+=100){
                for(Double nu = 0.01; nu<0.9; nu+=0.01){ // Variabile n smv, da 0 a 1 per nu-svc
                for(Double g = 0.00; g<0.01; g+=0.01){   
                for(int d = 1; d<2; d++){              

                    System.out.println(classificatorType+" ITR:"+iteration+" S:"+s+" K:"+k+" D:"+d+" N:"+n+" C:"+c+" NU:"+nu);

                    if(classificatorType.equals("tree")){
                        classificator = treeClassifier(train);
                    } else if(classificatorType.equals("svm")){
                        classificator = SVMClassifier(train, c, d, g, k, s, nu);
                    } else if(classificatorType.equals("knn")){
                        IBk ibk = new IBk();
                        String options = ( "-K 1"); // Set k
                        String[] optionsArray = options.split( " " );
                        ibk.setOptions( optionsArray );
                        classificator = ibk;
                    } else{
                        System.out.println("Nome classificatore errato");
                        exit(1);
                    }

                    Evaluation evalu = new Evaluation(train);
                    evalu.evaluateModel(classificator,train);
                    //System.out.println(evalu.toSummaryString("Results Training:\n", false));
                    // Evaluation on the test set
                    Evaluation evalTs = new Evaluation(train);
                    evalTs.evaluateModel(classificator,test);
                    /*System.out.println(evalTs.toSummaryString("Results Test:\n", false));*/
                    System.out.println(evalTs.toMatrixString());
                    System.out.println(evalTs.pctCorrect());
                    for(int i= 0;i< data.numClasses(); i++){
                        System.out.println("Sensitivity(TP/P) class "+i+": "+evalTs.truePositiveRate(i));
                        System.out.println("Specificity(TN/N) class "+i+": "+evalTs.trueNegativeRate(i));
                    }



                    accuracy.put(classificatorType+" "+iteration+" "+s+" "+k+" "+d+" "+n+" "+c+" "+nu,evalTs.pctCorrect());
                    classificatorAccuracy = evalTs.pctCorrect();
        }}}}}}
        
        
        accuracy.entrySet().forEach(entry->{
            System.out.println(entry.getKey() + " " + entry.getValue());  
        });
        
        return classificator;
        /*
        Stampa di prova
        for(Map.Entry<String, Double> entry: trainAttributes.entrySet()) {
            System.out.println(entry.getKey() + " --- " + entry.getValue());
        }*/   
    }



    
    }
