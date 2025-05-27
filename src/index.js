import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import modelRoutes from './routes/train.routes.js';
import modelController from './controllers/train.controller.js';
import fs from 'fs/promises';

const app = express();

// Crear directorios necesarios
try {
  await fs.access('./models');
} catch {
  await fs.mkdir('./models', { recursive: true });
}

try {
  await fs.access('./dataset');
} catch {
  await fs.mkdir('./dataset/frutas', { recursive: true });
  console.log(
    '📁 Carpeta dataset creada. Agrega imágenes en ./dataset/frutas/[nombre_fruta]/'
  );
}

// ✅ Inicializar modelo AL INICIO (solo una vez)
console.log('🤖 Inicializando modelo de IA...');
await modelController.inicializar();
console.log('✅ Inicialización completada');

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
  const info = modelController.getInfo();
  res.json({
    mensaje: 'API de Clasificación de Frutas',
    modeloDisponible: info.modeloEntrenado,
    clasesDisponibles: info.etiquetas,
    endpoints: [
      'POST /api/modelo/entrenar - Entrenar modelo con imágenes',
      'POST /api/modelo/predecir - Predecir clase de una imagen',
      'GET /api/modelo/info - Información del modelo',
      'GET /api/modelo/descargar - Descargar modelo entrenado',
    ],
  });
});

app.use('/api/modelo', modelRoutes);

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Servidor ejecutándose en puerto ${PORT}`);
  console.log(`📡 Endpoints disponibles en http://localhost:${PORT}`);
  console.log(`🌐 Interfaz web en http://localhost:${PORT}/index.html`);
});
