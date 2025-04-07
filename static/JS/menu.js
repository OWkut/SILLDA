function toggleMenu(side) {
  const menu = document.getElementById('menu' + (side === 'left' ? 'Left' : 'Right'));
  menu.classList.toggle('open');

  document.body.classList.toggle(`${side}-open`);
}

const videoTab = document.getElementById('videoTab');
const modalOverlay = document.getElementById('modalOverlay');
const closeModal = document.getElementById('closeModal');

videoTab.addEventListener('click', () => {
  modalOverlay.classList.add('active');
});

closeModal.addEventListener('click', () => {
  modalOverlay.classList.remove('active');
});

modalOverlay.addEventListener('click', (e) => {
  if (e.target === modalOverlay) {
    modalOverlay.classList.remove('active');
  }
});


function updatePerf() {
  fetch('/perf')
    .then(res => {
      if (!res.ok) throw new Error('Réponse serveur invalide');
      return res.json();
    })
    .then(data => {
      document.getElementById('fpsCurrent').textContent = data.current;
      document.getElementById('fpsMin').textContent = data.min;
      document.getElementById('fpsMax').textContent = data.max;
      document.getElementById('fpsAvg').textContent = data.avg;
    })
    .catch(err => {
      console.error('Erreur FPS:', err);
      document.getElementById('fpsCurrent').textContent = 'Err';
    });
}


setInterval(updatePerf, 1000);

function pauseStream() {
  fetch('/pause_stream', { method: 'POST' })
    .then(() => console.log('📡 Flux vidéo mis en pause'))
    .catch(err => console.error('Erreur pause:', err));
}

function resumeStream() {
  fetch('/resume_stream', { method: 'POST' })
    .then(() => console.log('▶️ Flux vidéo relancé'))
    .catch(err => console.error('Erreur reprise:', err));
}

function showTab(tabId) {
  const tabs = document.querySelectorAll('.tab-content');
  const buttons = document.querySelectorAll('.tab-btn');

  tabs.forEach(tab => {
    tab.style.display = tab.id === tabId ? 'block' : 'none';
  });

  buttons.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('onclick').includes(tabId));
  });
}
