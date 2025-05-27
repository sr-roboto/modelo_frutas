class ClasificadorFrutas {
  constructor() {
    this.modeloDisponible = false;
    this.etiquetas = [];
    this.verificarModelo();
    this.configurarEventos();
  }

  configurarEventos() {
    document
      .getElementById('imagen-clasificar')
      .addEventListener('change', (e) => {
        this.mostrarPreview(e.target.files[0]);
      });
  }

  mostrarPreview(archivo) {
    if (!archivo) return;

    const preview = document.getElementById('preview');
    const btnClasificar = document.getElementById('btn-clasificar');

    preview.src = URL.createObjectURL(archivo);
    preview.style.display = 'block';
    btnClasificar.disabled = !this.modeloDisponible;
  }

  mostrarResultado(elementId, mensaje, tipo = 'info') {
    const elemento = document.getElementById(elementId);
    elemento.textContent = mensaje;
    elemento.className = `resultado ${tipo}`;
  }

  async verificarModelo() {
    try {
      this.mostrarResultado(
        'info-modelo',
        'Verificando estado del modelo...',
        'loading'
      );

      const response = await fetch('/api/modelo/info');
      const data = await response.json();

      if (data.modeloEntrenado) {
        this.modeloDisponible = true;
        this.etiquetas = data.etiquetas;
        this.mostrarResultado(
          'info-modelo',
          `✅ Modelo entrenado con ${
            data.totalClases
          } clases: ${data.etiquetas.join(', ')}`,
          'success'
        );
      } else {
        this.modeloDisponible = false;
        this.mostrarResultado(
          'info-modelo',
          '❌ No hay modelo entrenado. Entrena uno primero.',
          'error'
        );
      }

      const btnClasificar = document.getElementById('btn-clasificar');
      btnClasificar.disabled = !this.modeloDisponible;
    } catch (error) {
      this.mostrarResultado(
        'info-modelo',
        '❌ Error al verificar el modelo: ' + error.message,
        'error'
      );
    }
  }

  async entrenarModelo() {
    const archivos = document.getElementById('entrenar-archivos').files;

    if (!archivos || archivos.length === 0) {
      this.mostrarResultado(
        'resultado-entrenamiento',
        '❌ Selecciona al menos una imagen para entrenar',
        'error'
      );
      return;
    }

    try {
      this.mostrarResultado(
        'resultado-entrenamiento',
        `📤 Subiendo ${archivos.length} imágenes...`,
        'loading'
      );

      const formData = new FormData();
      for (let archivo of archivos) {
        formData.append('imagenes', archivo);
      }

      this.mostrarResultado(
        'resultado-entrenamiento',
        '🧠 Entrenando modelo...',
        'loading'
      );

      const response = await fetch('/api/modelo/entrenar', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error en el servidor');
      }

      const resultado = await response.json();

      this.mostrarResultado(
        'resultado-entrenamiento',
        `✅ ${resultado.mensaje}. Clases: ${resultado.etiquetas.join(
          ', '
        )}. Total: ${resultado.totalImagenes} imágenes`,
        'success'
      );

      this.verificarModelo();
    } catch (error) {
      this.mostrarResultado(
        'resultado-entrenamiento',
        '❌ Error entrenando modelo: ' + error.message,
        'error'
      );
    }
  }

  async clasificarImagen() {
    const archivo = document.getElementById('imagen-clasificar').files[0];

    if (!archivo) {
      this.mostrarResultado(
        'resultado-clasificacion',
        '❌ Selecciona una imagen para clasificar',
        'error'
      );
      return;
    }

    try {
      this.mostrarResultado(
        'resultado-clasificacion',
        '🔍 Analizando imagen...',
        'loading'
      );

      const formData = new FormData();
      formData.append('imagen', archivo);

      const response = await fetch('/api/modelo/predecir', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error en el servidor');
      }

      const resultado = await response.json();

      this.mostrarResultado(
        'resultado-clasificacion',
        `🎯 Detectado: ${resultado.clase} (${(
          resultado.probabilidad * 100
        ).toFixed(2)}% de confianza)`,
        'success'
      );
    } catch (error) {
      this.mostrarResultado(
        'resultado-clasificacion',
        '❌ Error clasificando imagen: ' + error.message,
        'error'
      );
    }
  }
}

const clasificador = new ClasificadorFrutas();

function verificarModelo() {
  clasificador.verificarModelo();
}

function entrenarModelo() {
  clasificador.entrenarModelo();
}

function clasificarImagen() {
  clasificador.clasificarImagen();
}
