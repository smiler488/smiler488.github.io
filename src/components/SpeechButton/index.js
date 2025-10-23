import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

const SpeechButton = ({
  text,
  fallbackText,
  lang = 'zh-CN',
  rate = 0.8,
  pitch = 1.0,
  className = '',
  title = 'Click to hear pronunciation',
}) => {
  const [isSupported, setIsSupported] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);

  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      setIsSupported(true);
      
      const loadVoices = () => {
        const availableVoices = speechSynthesis.getVoices();
        setVoices(availableVoices);
        setIsReady(availableVoices.length > 0);
      };

      loadVoices();
      speechSynthesis.addEventListener('voiceschanged', loadVoices);
      
      return () => {
        speechSynthesis.removeEventListener('voiceschanged', loadVoices);
      };
    }
  }, []);

  const findBestVoice = (targetLang) => {
    if (!voices.length) return null;
    
    let voice = voices.find(v => v.lang === targetLang);
    if (voice) return voice;
    
    const langFamily = targetLang.split('-')[0];
    voice = voices.find(v => v.lang.startsWith(langFamily));
    if (voice) return voice;
    
    return voices.find(v => v.default) || voices[0];
  };

  const showTooltip = (message) => {
    const tooltip = document.createElement('div');
    tooltip.textContent = message;
    tooltip.className = styles.tooltip;
    document.body.appendChild(tooltip);
    
    const rect = document.activeElement?.getBoundingClientRect();
    if (rect) {
      tooltip.style.left = `${rect.left + rect.width / 2}px`;
      tooltip.style.top = `${rect.top - 40}px`;
    }
    
    setTimeout(() => {
      if (tooltip.parentNode) {
        tooltip.parentNode.removeChild(tooltip);
      }
    }, 2000);
  };

  const handleSpeak = async () => {
    if (!isSupported || !text) {
      const message = fallbackText || `Pronunciation: ${text}`;
      showTooltip(message);
      return;
    }

    if (!isReady) {
      showTooltip('Speech synthesis not ready yet');
      return;
    }

    try {
      speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      
      const voice = findBestVoice(lang);
      if (voice) {
        utterance.voice = voice;
      }
      
      utterance.lang = lang;
      utterance.rate = rate;
      utterance.pitch = pitch;
      utterance.volume = 1.0;
      
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event.error);
        setIsSpeaking(false);
        const message = fallbackText || `Pronunciation: ${text}`;
        showTooltip(message);
      };
      
      speechSynthesis.speak(utterance);
      
    } catch (error) {
      console.error('Speech error:', error);
      setIsSpeaking(false);
      const message = fallbackText || `Pronunciation: ${text}`;
      showTooltip(message);
    }
  };

  return (
    <button
      onClick={handleSpeak}
      className={`${styles.speechButton} ${className}`}
      title={title}
      disabled={isSpeaking}
      aria-label={`Pronounce ${text}`}
    >
      {isSpeaking ? '🔇' : '🔊'}
    </button>
  );
};

export default SpeechButton;