;(function(){
  try {
    var ok = false;
    if (typeof window !== 'undefined' && window.localStorage) {
      if (window.localStorage.getItem('app_session')) ok = true;
      for (var i = 0; i < window.localStorage.length; i++) {
        var k = window.localStorage.key(i) || '';
        if (k.startsWith('sb-') && k.endsWith('-auth-token')) { ok = true; break; }
      }
    }
    window.__APP_AUTH_OK__ = !!ok;
  } catch (e) {
    window.__APP_AUTH_OK__ = false;
  }
})();