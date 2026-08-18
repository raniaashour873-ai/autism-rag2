import {
    useEffect,
    useRef,
} from "react";

function useAutoScroll(dependencies = []) {

    const ref = useRef(null);

    useEffect(() => {

        const element =
            ref.current;

        if (!element) {
            return;
        }

        element.scrollTo({
            top: element.scrollHeight,
            behavior: "smooth",
        });

    }, dependencies);

    return ref;
}

export default useAutoScroll;