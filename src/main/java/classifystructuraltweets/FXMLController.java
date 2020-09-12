package classifystructuraltweets;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import utility.IOManager;
import static utility.IOManager.readFromCsvFile;
import static utility.IOManager.writeToCsvFile;
import java.net.URL;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import static javafx.collections.FXCollections.observableArrayList;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.DatePicker;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import utility.ClassifierData;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.StopwordsHandler;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class FXMLController implements Initializable {
    protected static ObservableList<TweetsTableView> tweetsOl = FXCollections.observableArrayList();
    @FXML TableView<TweetsTableView> tableTweets;  
    @FXML TableColumn<TweetsTableView, String> tweetCol;
    @FXML TableColumn<TweetsTableView, String> datCol;
    
    @FXML private TextField labelX, labelY, labelRange;
    @FXML Label waitingLabel;
    @FXML private TextArea textAreaTweet;
    @FXML private ComboBox comboClassifier;
    @FXML private DatePicker datePickerFrom, datePickerTo;
    @FXML public Button buttonSearch;
    @FXML public ImageView imagePlumbing;
    
    private ClassifierData cd; 
    
    
    private String position;    
    private List<String[]> tweets;
    private TweetsTableView actualRow;
    
    private static final String newDataset = "file/newDataset.csv";

    
    public void onActionButtonCorrect(ActionEvent event){
        exportTweet("Structural");
    }
    
    public void onActionButtonWrong(ActionEvent event){
        exportTweet("NonStructural");
    }
    
    private void exportTweet(String classLabel){
        tableTweets.getItems().remove(actualRow);
        textAreaTweet.clear();
        
        List<String[]> thingsToWrite = new ArrayList<>();
        String[] tweet = new String[3];
        
        tweet[0] = actualRow.getText();
        tweet[1] = classLabel;
        tweet[2] = actualRow.getWord();
                
        thingsToWrite.add(tweet);
        IOManager.writeAppendToCsvFile(thingsToWrite, ";", newDataset);
    }
    
    
    
    public void onActionButtonSearch(ActionEvent event){
        setClassifierCombobox();
        position = labelX.getText()+","+labelY.getText()+","+labelRange.getText()+"mi";
        
        System.out.println(datePickerFrom.getValue().toString());
        System.out.println(datePickerTo.getValue().toString());
        System.out.println(position);  
       

        buttonSearch.setText("Downloading tweets...");
        buttonSearch.setDisable(true);
 
        
        Platform.runLater(new Runnable(){
            @Override
            public void run() {
                String choosenClassifier = (String) comboClassifier.getValue();
                try {
                    cd = (ClassifierData) IOManager.loadClassBinary("classifier/"+choosenClassifier);
                } catch (IOException ex) {
                    ex.printStackTrace();
                } catch (ClassNotFoundException ex) {
                    ex.printStackTrace();
                }
                

                
                try {  
                    tweets = TweetsImport.getTweets(cd.researchKey, datePickerFrom.getValue().toString(), datePickerTo.getValue().toString(), position);
                } catch (ParseException ex) {
                    ex.printStackTrace();
                }
                
                // Rimuovo i tweet duplicati
                List<String[]> temp = new ArrayList<>();
                boolean insert = true;
                for(String[] s : tweets){
                    for(String[] sTemp: temp){
                       if(s[0].equals(sTemp[0])){ 
                           insert = false;
                           break;
                       }
                    }
                    if(insert){
                        temp.add(s);
                    }
                }
                
                tweets = temp;
                
                try {
                    //Classifico i tweet e rimuovo quelli inutili
                    classifyAndRemove();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
                      
                // Inserisco i tweet nella tabella
                tweetsOl.clear();
                for(String[] tweet:tweets){
                    try {
                        tweetsOl.add(new TweetsTableView(tweet[0],tweet[1],tweet[2],tweet[3]));
                    } catch (ParseException ex) {
                        Logger.getLogger(FXMLController.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
                                
                tableTweets.setItems(tweetsOl);  
                textAreaTweet.clear();
                
                buttonSearch.setText("SEARCH");
                buttonSearch.setDisable(false);            
            }
        });
        
    }
    
    private void classifyAndRemove() throws Exception{
        String[] unlabeledTweets = new String[tweets.size()];
        
        for(int i=0; i<tweets.size(); i++){
            String[] t = tweets.get(i);
            unlabeledTweets[i] = t[0];
        }
        
        //Preprocessing
        unlabeledTweets = deleteUrlAndPic(unlabeledTweets);    

        List<String[]> out = new ArrayList<>();
        String[] tmp = new String[2];
        
        tmp[0] = "tweet";
        tmp[1] = "classe";
        out.add(tmp);
        
        for(int pos = 0; pos<tweets.size(); pos++){
            tmp = new String[2];
            tmp[0] = "'"+unlabeledTweets[pos]+"'";
            tmp[1] = "n";
            out.add(tmp);
        }
      
        writeToCsvFile(out,",","tmp.csv");
        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("tmp.csv");

        Instances data = source.getDataSet();
        
        // Converto il tipo dei tweets da nominal a string cosi da poter appplicare StringToWordVector
        String[] nsOptions = {"-C", "1"};
        NominalToString nsFilter = new NominalToString();
        nsFilter.setInputFormat(data);
        nsFilter.setOptions(nsOptions);
        data = Filter.useFilter(data, nsFilter);
        
        // Selezione Class come attributo classe
        data.setClassIndex(data.numAttributes() - 1);  
    
     
        //data.setClass(data.attribute("classLabel"));
        Instances labeled = new Instances(data);   
        System.out.println("Totale tweet analizzati ---> "+labeled.numInstances());
        
        
// label instances
        double clsLabel;
        for (int i = 0; i < data.numInstances(); i++) {
           clsLabel = cd.classifier.classifyInstance(data.instance(i));
           labeled.instance(i).setClassValue(clsLabel);
        }

        //Adesso elimino quelli che non c'entrano niente dal dataset dda mostrare        
        ArrayList<String[]> tweetsToShow = new ArrayList<>();
        double[] instancesClass = labeled.attributeToDoubleArray(labeled.numAttributes()-1);
        for(int i=0; i<tweets.size(); i++){
            
            if(instancesClass[i] == 1.0){ // 1.0 Strutturali, 0.0 Non Strutturali
                tweetsToShow.add(tweets.get(i));
               
            }
        }
        System.out.println("Totale tweet classificati strutturali ---> "+tweetsToShow.size());
        waitingLabel.setVisible(true);
        waitingLabel.setText("Selected "+tweetsToShow.size()+ " out of "+ labeled.numInstances() + " tweets downloaded");
        tweets = tweetsToShow;
    }
    
    public void setClassifierCombobox(){
        //Estraggo i nomi dei classificatori nella folder Classifier
        File folder = new File("classifier");
        File[] listOfFiles = folder.listFiles();
        String[] names = new String[listOfFiles.length];
        for (int i = 0; i < listOfFiles.length; i++) {
          if (listOfFiles[i].isFile()) {
            names[i] = listOfFiles[i].getName();
          }
        }
        comboClassifier.setItems(FXCollections.observableArrayList(names));
    }
    
    
    
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        setClassifierCombobox();
        tableTweets.setItems(tweetsOl);  
        
        try {
            imagePlumbing.setImage(new Image(new FileInputStream("plumbing.png")));
        } catch (FileNotFoundException ex) {
            System.out.println("[ERROR] Immagine non trovata");
        }
        
        // Coordinate di Pisa settate di default
        labelX.setText("43.7118");
        labelY.setText("10.4147");
        labelRange.setText("3"); // Circa 5km
        
        // Specifico le colonne cosa devono mostrare 
        tweetCol.setCellValueFactory(new PropertyValueFactory<>("text"));
        datCol.setCellValueFactory(new PropertyValueFactory<>("date"));
        
        tableTweets.setRowFactory( tv ->{
            TableRow<TweetsTableView> row = new TableRow<>();
                        
            row.setOnMouseClicked((Event e) ->{
                if(!row.isEmpty()){                    
                    actualRow = row.getItem();
                    textAreaTweet.setText(actualRow.getText());
                }                          
            });
                      
            return row;
        });
       


    } 
    
    private static String[] deleteUrlAndPic(String[] tweets){
        String urlRegex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+";
        String picRegex = "pic.twitter.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+";
        String hashtagRegex = "\\B#\\w*[a-zA-Z]+\\w*";
        String mentionsRegex = "(?:@[\\w_]+)";        
   
        for(int i = 0; i<tweets.length; i++){
            // "\\p{P}" rimuove tutta la punteggiatura, "( )+" rimpiazza spazzi bianchi multipli con uno solo 
            tweets[i] = tweets[i].replaceAll(urlRegex, "").replaceAll(picRegex, "").replaceAll(hashtagRegex, "")
                            .replaceAll(mentionsRegex, "").replaceAll("\\p{P}", " ").replaceAll("( )+"," ").toLowerCase();
        }  
        return tweets;
    }
}
