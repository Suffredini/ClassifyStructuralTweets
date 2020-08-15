package classifystructuraltweets;

import utility.IOManager;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import utility.ExportedClassifierData;


public class MainApp extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        Parent root = FXMLLoader.load(getClass().getResource("/fxml/Scene.fxml"));
        
        Scene scene = new Scene(root);
        scene.getStylesheets().add("/styles/Styles.css");
        
        stage.setTitle("Search structural tweets");
        stage.setScene(scene);
        stage.show();
    }


    public static void main(String[] args) throws IOException, FileNotFoundException, ClassNotFoundException {        
        launch(args);
    }

}
