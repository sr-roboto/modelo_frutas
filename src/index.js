import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import modelRoutes from './routes/train.routes.js';
import fs from 'fs/promises';

const app = express();

// Crear directorio para modelos si no existe
try {
  await fs.access('./models');
} catch {
  await fs.mkdir('./models', { recursive: true });
}

// Middleware
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
  res.json({
    mensaje: 'API de Clasificación de Frutas',
    endpoints: [
      'POST /api/modelo/entrenar - Entrenar modelo con imágenes',
      'POST /api/modelo/predecir - Predecir clase de una imagen',
      'GET /api/modelo/info - Información del modelo',
    ],
  });
});

app.use('/api/modelo', modelRoutes);

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Servidor ejecutándose en puerto ${PORT}`);
  console.log(`📡 Endpoints disponibles en http://localhost:${PORT}`);
});
