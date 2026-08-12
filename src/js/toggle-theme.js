document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('theme-toggle');

  if (!toggleBtn) return;

  function updateButtonText(theme) {
    toggleBtn.textContent = theme === 'dark' ? 'LIGHT' : 'DARK';
  }

  const currentTheme = document.documentElement.getAttribute('data-theme');
  updateButtonText(currentTheme);

  toggleBtn.addEventListener('click', () => {
    const activeTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = activeTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateButtonText(newTheme);
  });
});
