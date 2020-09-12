package classifystructuraltweets;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import javafx.beans.property.SimpleStringProperty;

public class TweetsTableView {
    private final SimpleStringProperty text;  
    private final SimpleStringProperty date; 
    private final SimpleStringProperty position; 
    private final SimpleStringProperty word; 

    public TweetsTableView(String t, String d, String p, String w) throws ParseException{        
        text = new SimpleStringProperty(t);
        
        String[]dateTmp = d.split(" ");
        
        date = new SimpleStringProperty(dateTmp[2]+" "+dateTmp[1]+" "+dateTmp[5]);
        position = new SimpleStringProperty(p);   
        word = new SimpleStringProperty(w); 
    }

    public String getText(){
       return text.get();
    } 
    
    public void setText(String s){
        text.set(s);
    }
    
    public String getDate(){
       return date.get();
    }
    
    public void setDate(String n){
       date.set(n);
    }
    
    public String getPosition(){
       return position.get();
    } 
    
    public void setPosition(String s){
        position.set(s);
    } 
    
    public String getWord(){
       return word.get();
    }
    
    public void setWord(String t){
       word.set(t);
    }

}