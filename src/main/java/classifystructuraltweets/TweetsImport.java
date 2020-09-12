package classifystructuraltweets;

import java.text.DateFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class TweetsImport {
    public static List<String[]> getTweets(String[] words, String fromDate, String toDate, String position) throws ParseException {
        TwitterCriteria criteria = null;
        Tweet t = null;
        List<String[]> allTweets = new ArrayList<String[]>();
        for(String word : words){
            System.out.println(word);
            System.out.println(fromDate);
            criteria = TwitterCriteria.create()
                .setQuerySearch(word)
                .setSince(fromDate)
                .setUntil(toDate)
                .setPosition(position);                    
                       
            for (Tweet tw : TweetManager.getTweets(criteria)) {	               
                allTweets.add(new String[]{tw.getText(), tw.getDate().toString(), tw.getGeo(), word});    
            }
        }            
        return allTweets;  
    }        
}

