/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utility;

import java.io.*;
import java.util.*;


/**
 *
 * @author suffr
 */
public class IOManager {
    
    public static Object loadClassBinary(String name) throws FileNotFoundException, IOException, ClassNotFoundException{
        FileInputStream fos = new FileInputStream(name);
        ObjectInputStream oos = new ObjectInputStream(fos);
        Object o = oos.readObject();
        oos.close();
        return o;
    }
    
    public static void saveClassBinary(String name, Object o) throws FileNotFoundException, IOException{
        FileOutputStream fos = new FileOutputStream(name);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(o);
        oos.close();
    }
    
    public static void printListString(List<List<String>> lls){
        for(int i = 0; i<lls.size(); i++){
            List<String> ls = lls.get(i);
            String print = "";
            for(int j = 0; j<ls.size(); j++){
                print += ls.get(j) + "-";
            }
            System.out.println(print);
        } 
    }
    
    public static List<String> readFromFile(String fileName){
        List<String> list = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))){
            String line = "";
            while((line = reader.readLine()) != null){
                list.add(line.replace(" ", ""));
            }
            return list;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }  
    }
    
    public static List<String[]> readFromCsvFile(String separator, String fileName){
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))){
            List<String[]> list = new ArrayList<>();
            String line = "";
            while((line = reader.readLine()) != null){
                String[] array = line.split(separator);
                list.add(array);
            }
            return list;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }  
    }
    
    
    public static void writeToCsvFile(List<String[]> thingsToWrite, String separator, String fileName){
        try (FileWriter writer = new FileWriter(fileName)){
            for (String[] strings : thingsToWrite) {
                for (int i = 0; i < strings.length; i++) {
                    writer.append(strings[i]);
                    if(i < (strings.length-1))
                        writer.append(separator);
                }
                writer.append(System.lineSeparator());
            }
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    
}
