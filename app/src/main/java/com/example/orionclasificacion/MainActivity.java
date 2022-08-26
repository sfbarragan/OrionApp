package com.example.orionclasificacion;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.orionclasificacion.ml.ModelV7;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


/**
 * Esta clase se ejecuta por defecto al abrir la aplicación, cargando los componentes necesarios
 * <p>
 * Este método permitira funcionar correctamente al programa cuando el usuario realice alguna acción
 * o interactue con el usuario.
 *
 */
public class MainActivity extends AppCompatActivity {


    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 227;


    /**
     * Ejecución de acciones.
     *
     * <p>Usa la instancia guardada para ejecutar las acciones solicitadas por el usuario.
     *
     * @param savedInstanceState instancia de ejecución del programa
     * @since             1.0
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        /* Creamos la interfaz de la aplicación*/
        super.onCreate(savedInstanceState);
        /* Cargamos los elementos del layout */
        setContentView(R.layout.activity_main);

        /* Cargamos el botón que accionara la camara */
        camera = findViewById(R.id.button);
        /* Cargamos el botón que abrira la galeria */
        gallery = findViewById(R.id.button2);

        /* Cargamos la etiqueta que mostrara el mensaje */
        result = findViewById(R.id.result);
        /* Cargamos el contenedor que mostrara la imagen de la persona*/
        imageView = findViewById(R.id.imageView);

        /* En caso de accionar la camara*/
        camera.setOnClickListener(new View.OnClickListener() {
            /**
             * Solicita permisos y permite la utilización de la camara del teléfono
             *
             * <p>Genera una nueva vista.
             *
             * @param view vista generada por el telefono para abrir la camara
             * @since             1.0
             */
            @Override
            public void onClick(View view) {
                /* Comprobamos la versión del SDK*/
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    /* En caso de que se consedan los permisos*/
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                       /* Inicia la camara */
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        /* Acciona la camara */
                        startActivityForResult(cameraIntent, 3);
                    } else {
                        /* Muestra un mensaje con la denegación de acceso */
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                    }
                }
            }
        });

        /* En caso de accionar la galeria*/
        gallery.setOnClickListener(new View.OnClickListener() {

            /**
             * Ejecuta la galeria del telefono
             *
             * <p>Genera una nueva vista.
             *
             * @param view vista generada por el telefono para abrir la galeria
             * @since             1.0
             */
            @Override
            public void onClick(View view) {
                /* Ejecutamos la galeria*/
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                /* Se inicia la galeria*/
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    /**
     * Este método permite clasificar a las personas.
     *
     * <p>Usa el mapa de bits de las imagenes para ejecutar la clasificación.
     *
     * @param image imapa de bits de las imagenes a clasificar
     * @since             1.0
     */
    public void classifyImage(Bitmap image){


        try {
            /* Instanciamos el modelo */
            ModelV7 model = ModelV7.newInstance(getApplicationContext());

            // Creamos los imputs de referencia
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 227, 227, 3}, DataType.FLOAT32);
            /* Redimencionamos el buffer de las imagenes*/
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4* imageSize* imageSize*3);
            /* Ordenamos el buffer de las imagenes*/
            byteBuffer.order(ByteOrder.nativeOrder());

            /* Inicalizamos los valores */
            int[] intValues = new int[imageSize * imageSize];
            /* Agrupamos el mapa de bits */
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            /* Inicializamos el número de pixeles*/
            int pixel = 0;
            //itere sobre cada píxel y extraiga los valores R, G y B. Agregue esos valores individualmente al búfer de bytes.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            //Almacenamos el buffer de bits
            inputFeature0.loadBuffer(byteBuffer);

            // Ejecuta la inferencia del modelo y obtiene el resultado..
            ModelV7.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Almacenamos el resultado de la inferencia
            float[] confidences = outputFeature0.getFloatArray();
            // Inicializamos la posición maxima
            int maxPos = 0;
            // Inicializamos la confianza maxima
            float maxConfidence = 0;
            // Encuentre el índice de la clase con mayor confianza.
            for (int i = 0; i < confidences.length; i++) {
                // Comprobamos la confiabililidad de los resultados
                if (confidences[i] > maxConfidence) {
                    //Almacenamos el resultado con mas confiaza
                    maxConfidence = confidences[i];
                    // Almacenamos la posición
                    maxPos = i;
                }
            }
            // Creamos un arreglo con las clases
            String[] classes = {"Ariel Chabla","Arrobo Mercy", "Steven Barragan", "Cevallos Joan", "Enriquez Selena", "Genesis Heredia"
                    ,"Goyes Anthony", "Hector Cedeño", "Jhon Zambrano", "Jordan Espinosa", "Jorge Borrero", "Jose Ruiz",
                    "Lucio Carlos", "Masache Fernando", "Melany López", "Mosquera Lucy", "Nataly Acosta",
                    "Olalla Luis", "Parraga Maria Jose", "Paute Kevin", "Raymond Davila", "Rivas Selena", "Salazar Johana",
                    "Solano Wilmer", "Solorzano Bryan", "Vinicio Borja"};

            // Mostramos la clase resultado
            result.setText(classes[maxPos]);

            // Libera los recursos del modelo si ya no se usan
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    /**
     * Este método permite ejecutar acciónes en base a las solicitudes del usuario.
     *
     * <p> Usa el código de respuesta, el código de resultado y la data.
     *
     * @param requestCode código de respuesta
     * @param resultCode código de resultado
     * @param data datos de las imagenes
     * @since             1.0
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        // En caso de que se ejecute el resultado sin errores
        if(resultCode == RESULT_OK){
            // En caso de que el código de respuesta sea igual a 3
            if(requestCode == 3){
                // Cargamos el mapa de bits
                Bitmap image = (Bitmap) data.getExtras().get("data");
                // Cargamos las dimensiones de la imagen
                int dimension = Math.min(image.getWidth(), image.getHeight());
                // Generamos la imagen
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                // Mostramos la imagen a través de su mapa de bits
                imageView.setImageBitmap(image);

                // Escalamos la imagen desde el mapa de bits
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                // Enviamos la imagen a clasificar
                classifyImage(image);
            }else{
                // Cargamos la data
                Uri dat = data.getData();
                // Inicializamos el mapa de bit de la imagene
                Bitmap image = null;
                try {
                    // Cargamos el contenedor sin imagen
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    // En caso de error
                    e.printStackTrace();
                }
                // Modificamos el mapa de bits
                imageView.setImageBitmap(image);

                // Generamos la escal del mapa de bit de la imagen
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                // Enviamos la imagen a clasificar
                classifyImage(image);
            }
        }
        // Enviamos a través de result el requestCode, resulCode y la data
        super.onActivityResult(requestCode, resultCode, data);
    }
}
