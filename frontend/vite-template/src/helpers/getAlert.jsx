import React from 'react';
import ReactDOM from 'react-dom';
import AlertInform from '../components/alert/Alert';

const getAlert = (title, text, containerId) => {
  const container = document.getElementById(containerId); // Находим контейнер
  if (!container) return;

  console.log('ALERT');
  const AlertComponent = () => (
    <AlertInform
      title={title + "ALERT"}
      text={text}
      onClose={() => {
        // eslint-disable-next-line react/no-deprecated
        ReactDOM.unmountComponentAtNode(container); // Удаляем компонент
      }}
    />
  );

  // eslint-disable-next-line react/no-deprecated
  ReactDOM.render(<AlertComponent />, container); // Рендерим компонент в контейнер
};

export default getAlert;