import React from 'react';

import LyricList from '../components/LyricsList'
import {Container, Col} from 'react-bootstrap'

export default class IndexPage extends React.Component {
    render() {
        return (
            <Container>
                <br></br>
                <Col>
                    <LyricList/>
                </Col>
            </Container>
        )
    }
}