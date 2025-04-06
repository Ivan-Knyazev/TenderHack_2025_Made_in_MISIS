import { Card, CardBody } from "@heroui/card";


const LeftCard = ({text, ...props}: {text: string}) => {
    return(
        <Card
        shadow={'none'}
        isHoverable={true}
        isPressable={true}
        >
          <CardBody>
            <p className="text-[14px] !font-normal mt-[-16px] mb-[-16px]">{text}</p>
          </CardBody>
        </Card>
      );
}
export default LeftCard;