import {useState} from 'react';
import {Button} from '@heroui/button';
import {Alert} from "@heroui/alert";

const AlertInform = ({title, text, ...props}) => {
const [isVisible, setIsVisible] = useState(true);

  return (
    <div className="flex flex-col gap-4">
      {isVisible ? (
        <Alert
          color="success"
          description={text}
          isVisible={isVisible}
          title={title + 'ALERT'}
          variant="faded"
          onClose={() => setIsVisible(false)}
        />
      ) : (
        <></>
      )}
    </div>
  );
}

export default AlertInform;