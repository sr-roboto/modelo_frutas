class ClasificadorFrutas {
  constructor() {
    this.modeloDisponible = false;
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
      const response = await fetch('/api/modelo/info');
      const data = await response.json();

      if (data.modeloEntrenado) {
        this.modeloDisponible = true;
        this.mostrarResultado(
          'info-modelo',
          `✅ Modelo listo. Clases disponibles: ${data.etiquetas.join(', ')}`,
          'success'
        );

        // Habilitar botón de descarga
        document.getElementById('btn-descargar').disabled = false;
      } else {
        this.modeloDisponible = false;
        this.mostrarResultado(
          'info-modelo',
          '❌ No hay modelo disponible',
          'error'
        );

        // Deshabilitar botón de descarga
        document.getElementById('btn-descargar').disabled = true;
      }

      document.getElementById('btn-clasificar').disabled =
        !this.modeloDisponible;
    } catch (error) {
      this.mostrarResultado(
        'info-modelo',
        '❌ Error conectando con el servidor',
        'error'
      );
    }
  }

  async clasificarImagen() {
    const archivo = document.getElementById('imagen-clasificar').files[0];

    if (!archivo) {
      this.mostrarResultado(
        'resultado-clasificacion',
        '❌ Selecciona una imagen',
        'error'
      );
      return;
    }

    try {
      this.mostrarResultado(
        'resultado-clasificacion',
        '🔍 Analizando...',
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
        throw new Error(errorData.error);
      }

      const resultado = await response.json();

      this.mostrarResultado(
        'resultado-clasificacion',
        `🎯 ${resultado.clase} (${(resultado.probabilidad * 100).toFixed(
          1
        )}% de confianza)`,
        'success'
      );
    } catch (error) {
      this.mostrarResultado(
        'resultado-clasificacion',
        '❌ Error: ' + error.message,
        'error'
      );
    }
  }

  async descargarModelo() {
    try {
      // Mostrar estado de descarga
      const btnDescargar = document.getElementById('btn-descargar');
      const textoOriginal = btnDescargar.textContent;

      btnDescargar.textContent = '⏳ Descargando...';
      btnDescargar.disabled = true;

      // Realizar descarga
      const response = await fetch('/api/modelo/descargar');

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error);
      }

      // Obtener el blob del archivo ZIP
      const blob = await response.blob();

      // Crear URL temporal para la descarga
      const url = window.URL.createObjectURL(blob);

      // Crear elemento de descarga temporal
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'modelo-frutas.zip';

      // Agregar al DOM, hacer clic y remover
      document.body.appendChild(a);
      a.click();

      // Limpiar recursos
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      // Mostrar mensaje de éxito
      this.mostrarResultado(
        'info-modelo',
        '✅ Modelo descargado exitosamente como modelo-frutas.zip',
        'success'
      );

      // Restaurar botón
      btnDescargar.textContent = textoOriginal;
      btnDescargar.disabled = false;
    } catch (error) {
      // Mostrar error
      this.mostrarResultado(
        'info-modelo',
        '❌ Error descargando modelo: ' + error.message,
        'error'
      );

      // Restaurar botón
      const btnDescargar = document.getElementById('btn-descargar');
      btnDescargar.textContent = '📥 Descargar Modelo';
      btnDescargar.disabled = false;
    }
  }
}

const clasificador = new ClasificadorFrutas();

function verificarModelo() {
  clasificador.verificarModelo();
}

function clasificarImagen() {
  clasificador.clasificarImagen();
}

function descargarModelo() {
  clasificador.descargarModelo();
}
