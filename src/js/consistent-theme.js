(function() {
  const savedTheme = localStorage.getItem('theme');
  // const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  // if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();
